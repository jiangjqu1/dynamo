// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Devbar Storage Support
//!
//! BAR-mapped (C2C / GB200-class) device memory exposed host-side, used as the
//! KVBM tier-2 (host-tier) KV offloading backend.
//!
//! When no devbar region is configured (`DYN_KVBM_DEVBAR_BDF` or
//! `DYN_KVBM_DEVBAR_MOCK` unset), [`DevbarAllocator`] falls back to standard
//! pinned host allocation so default deployments are unaffected.
//!
//! ## NIXL registration
//!
//! [`DevbarStorage`] implements [`NixlDescriptor`] following
//! `DYN_KVBM_DEVBAR_MEM_TYPE` for BAR-backed instances — auto/dram (default;
//! the 2026-10-04 evidence: a Vram registration of the raw sysfs mmap fails at
//! ucp_mem_map with EIO on PCIe) or vram for peer-mapped products — with the
//! configured device ID, so the serialized descriptor carries the
//! "Device ID + memory type" triple required by the KVBM design doc's
//! metadata-exchange section. The
//! mock region and the pinned fallback report `MemType::Dram` — both are host
//! memory, and a Vram registration of host memory fails in UCX's cuda
//! transport on real-GPU hosts. The Vmm backing reports `MemType::Vram` at a
//! real device VA — semantically correct on every system class (it *is*
//! device memory), so it needs no per-class probe.

use std::{
    path::{Path, PathBuf},
    ptr::NonNull,
};

use super::nixl::{MemType, MemoryRegion, NixlAccessible, NixlDescriptor, NixlRegisterableStorage};
use super::{
    CudaAccessible, Local, RegistationHandle, RegisterableStorage, RegistrationHandles, Storage,
    StorageAllocator, StorageError, StorageMemset, StorageType, SystemAccessible,
};
use cudarc::driver::sys;
use dynamo_runtime::config::environment_names::kvbm::devbar as env_devbar;

/// How the devbar region is provisioned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DevbarBacking {
    /// BAR-mapped device memory, host-accessible (real devbar, e.g. GB200/GB300 C2C).
    Bar,
    /// Anonymous host mapping simulating a devbar region (plumbing tests without hardware).
    Mock,
    /// Standard pinned host allocation (fallback when no devbar region is configured).
    Pinned,
    /// CUDA-VMM device-alias: cuMemCreate'd physical VRAM mapped at a reserved
    /// GPU VA. Driver-owned (collision-free by construction), native D2D
    /// transfers; `ptr` is a DEVICE pointer — no CPU access exists.
    Vmm,
}

/// BAR-mapped device memory storage for the KVBM tier-2 (host-tier) offloading backend.
///
/// Mirrors [`super::PinnedStorage`] but is backed by a host-mapped BAR aperture
/// (`/sys/bus/pci/devices/<bdf>/resource{N}` mmap, auto-selected or explicit),
/// an anonymous mock mapping, or — when no devbar region is configured —
/// standard pinned allocation.
#[derive(Debug)]
pub struct DevbarStorage {
    ptr: NonNull<u8>,
    len: usize,
    device_id: u32,
    backing: DevbarBacking,
    handles: RegistrationHandles,
    /// Present only in Pinned fallback mode; owns the pinned allocation.
    pinned: Option<dynamo_memory::PinnedStorage>,
    /// Present only in Vmm mode; owns the CUDA-VMM allocation and the context
    /// needed for device-side operations (memset, teardown).
    vmm: Option<VmmOwn>,
    /// Present only in Bar mode with the IOMEMORY registration
    /// (`DYN_KVBM_DEVBAR_TRANSFERS=direct`); owns the registration and the
    /// window's device VA for direct GPU-side transfers.
    io: Option<IoOwn>,
}

/// Owns the VMM physical handle, the reserved device VA, and the CUDA context
/// of a Vmm-backed devbar region.
#[derive(Debug)]
struct VmmOwn {
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    handle: cudarc::driver::sys::CUmemGenericAllocationHandle,
    va: cudarc::driver::sys::CUdeviceptr,
    size: usize,
}

/// Owns the IOMEMORY registration and the device VA of a Bar-backed devbar
/// region registered for direct transfers (`DYN_KVBM_DEVBAR_TRANSFERS=direct`).
#[derive(Debug)]
struct IoOwn {
    /// The CUDA context the registration is bound to (the device's primary
    /// context); the Drop binds it before unregistering (the Vmm Drop's lesson).
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    /// The device VA `cuMemHostGetDevicePointer` returned for the window mmap.
    dev_va: cudarc::driver::sys::CUdeviceptr,
}

// SAFETY: DevbarStorage owns its mapping exclusively; raw-pointer access is
// confined to the `Storage`/`StorageMemset` unsafe APIs, mirroring SystemStorage.
unsafe impl Send for DevbarStorage {}
unsafe impl Sync for DevbarStorage {}

impl Local for DevbarStorage {}
impl SystemAccessible for DevbarStorage {}
impl CudaAccessible for DevbarStorage {}

fn curesult(r: cudarc::driver::sys::CUresult, what: &str) -> Result<(), StorageError> {
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        Err(StorageError::AllocationFailed(format!(
            "{what} failed: {r:?}"
        )))
    } else {
        Ok(())
    }
}

impl DevbarStorage {
    /// Create a devbar-backed storage by mmapping `size` bytes at `offset` from a
    /// BAR aperture device file (e.g. `/sys/bus/pci/devices/<bdf>/resource0`).
    pub fn from_bar(
        path: PathBuf,
        offset: u64,
        size: usize,
        device_id: u32,
    ) -> Result<Self, StorageError> {
        let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| StorageError::InvalidConfig(e.to_string()))?;
        let fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_RDWR | libc::O_SYNC | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(StorageError::AllocationFailed(format!(
                "failed to open devbar region file {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }

        // The sysfs resource file's length is the aperture size.
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut stat) } != 0 {
            unsafe { libc::close(fd) };
            return Err(StorageError::AllocationFailed(format!(
                "fstat failed for {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        let aperture_size = stat.st_size as u64;
        if !offset.is_multiple_of(unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64) {
            unsafe { libc::close(fd) };
            return Err(StorageError::InvalidConfig(format!(
                "devbar offset {offset} is not page-aligned; sysfs resource mmap of {} \
                 requires a page-aligned offset.",
                path.display()
            )));
        }
        let Some(request_end) = offset.checked_add(size as u64) else {
            unsafe { libc::close(fd) };
            return Err(StorageError::InvalidConfig(format!(
                "devbar request (offset {offset} + {size} bytes) overflows u64; \
                 cannot mmap {}.",
                path.display()
            )));
        };
        if request_end > aperture_size {
            unsafe { libc::close(fd) };
            return Err(StorageError::InvalidConfig(format!(
                "devbar request (offset {offset} + {size} bytes) exceeds aperture size \
                 ({aperture_size} bytes) of {}. KVBM tier-2 capacity must be >= the GPU KV \
                 cache size; reduce DYN_KVBM_CPU_CACHE_GB.",
                path.display()
            )));
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                offset as libc::off_t,
            )
        };
        unsafe { libc::close(fd) };
        if ptr == libc::MAP_FAILED {
            return Err(StorageError::AllocationFailed(format!(
                "mmap of devbar region {} failed: {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        let ptr = NonNull::new(ptr as *mut u8)
            .ok_or_else(|| StorageError::AllocationFailed("mmap returned null".into()))?;

        // Direct transfers (`DYN_KVBM_DEVBAR_TRANSFERS=direct`): register the
        // window with CUDA as I/O memory and take its device VA — the
        // bar-direct-spike-verified path (~3.0 GiB/s both directions on cm008).
        // The registration requires a privileged process (root or a
        // CAP_SYS_ADMIN-equivalent): the documented non-privileged failure is
        // CUDA_ERROR_NOT_PERMITTED. Clean error — the sticky-context lesson
        // covers copies, not registrations.
        let io = if std::env::var(env_devbar::DYN_KVBM_DEVBAR_TRANSFERS).as_deref() == Ok("direct")
        {
            let ctx = cudarc::driver::CudaContext::new(device_id as usize).map_err(|e| {
                StorageError::InvalidConfig(format!(
                    "direct transfers need a CUDA context on device {device_id}: {e}"
                ))
            })?;
            ctx.bind_to_thread()
                .map_err(|e| StorageError::InvalidConfig(format!("cuda bind: {e}")))?;
            let r = unsafe {
                sys::cuMemHostRegister_v2(
                    ptr.as_ptr() as *mut core::ffi::c_void,
                    size,
                    sys::CU_MEMHOSTREGISTER_IOMEMORY | sys::CU_MEMHOSTREGISTER_DEVICEMAP,
                )
            };
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(StorageError::InvalidConfig(format!(
                    "cuMemHostRegister(IOMEMORY|DEVICEMAP) failed: {r:?} — direct BAR transfers \
                     require a privileged process (root or CAP_SYS_ADMIN-equivalent; the \
                     documented non-privileged failure is CUDA_ERROR_NOT_PERMITTED). \
                     Use the default staged transfers or run the worker privileged."
                )));
            }
            let mut dev_va: sys::CUdeviceptr = 0;
            let rg = unsafe {
                sys::cuMemHostGetDevicePointer_v2(
                    &mut dev_va,
                    ptr.as_ptr() as *mut core::ffi::c_void,
                    0,
                )
            };
            curesult(rg, "cuMemHostGetDevicePointer")?;
            Some(IoOwn { ctx, dev_va })
        } else {
            None
        };

        Ok(Self {
            ptr,
            len: size,
            device_id,
            backing: DevbarBacking::Bar,
            handles: RegistrationHandles::new(),
            pinned: None,
            vmm: None,
            io,
        })
    }

    /// The io mapping for direct transfers: (host base, device base, length) —
    /// the window's host VA, its registered device VA, and the window length
    /// (the executor's per-block range check). Present only when the IOMEMORY
    /// registration succeeded (`TRANSFERS=direct`).
    pub fn io_mapping(&self) -> Option<(usize, u64, usize)> {
        self.io
            .as_ref()
            .map(|io| (self.ptr.as_ptr() as usize, io.dev_va, self.len))
    }

    /// Create a mock devbar-backed storage from an anonymous mapping.
    /// Exercises the devbar plumbing on machines without BAR-mapped device memory.
    pub fn mock(size: usize, device_id: u32) -> Result<Self, StorageError> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(StorageError::AllocationFailed(format!(
                "anonymous mmap failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        let ptr = NonNull::new(ptr as *mut u8)
            .ok_or_else(|| StorageError::AllocationFailed("mmap returned null".into()))?;
        Ok(Self {
            ptr,
            len: size,
            device_id,
            backing: DevbarBacking::Mock,
            handles: RegistrationHandles::new(),
            pinned: None,
            vmm: None,
            io: None,
        })
    }

    /// Create a devbar storage backed by standard pinned host allocation (fallback).
    pub fn pinned_fallback(size: usize, device_id: Option<u32>) -> Result<Self, StorageError> {
        let mut inner = dynamo_memory::PinnedStorage::new_for_device(size, device_id)?;
        let ptr = NonNull::new(unsafe { inner.as_mut_ptr() }).ok_or_else(|| {
            StorageError::AllocationFailed("pinned allocation returned null".into())
        })?;
        Ok(Self {
            ptr,
            len: size,
            device_id: device_id.unwrap_or(0),
            backing: DevbarBacking::Pinned,
            handles: RegistrationHandles::new(),
            pinned: Some(inner),
            vmm: None,
            io: None,
        })
    }

    /// Create a devbar-backed storage from a CUDA-VMM device-alias allocation.
    ///
    /// `cuMemCreate` reserves physical VRAM that no other cudaMalloc can hand
    /// out (collision-free by construction); `cuMemMap` maps it at a reserved
    /// GPU VA. The storage's pointer is a DEVICE pointer — transfers must use
    /// the D2D strategy (see `TransferContext::host_transfer_adjust`) and
    /// memset goes device-side; there is no CPU access. Works on C2C and
    /// PCIe GPUs.
    pub fn from_vmm(size: usize, device_id: u32) -> Result<Self, StorageError> {
        if size == 0 {
            return Err(StorageError::InvalidConfig("vmm size must be > 0".into()));
        }
        let ctx = cudarc::driver::CudaContext::new(device_id as usize).map_err(|e| {
            StorageError::AllocationFailed(format!("CudaContext::new({device_id}) failed: {e}"))
        })?;
        ctx.bind_to_thread()
            .map_err(|e| StorageError::AllocationFailed(format!("bind_to_thread failed: {e}")))?;

        let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.__bindgen_anon_1.id = device_id as core::ffi::c_int;

        let mut granularity: usize = 0;
        curesult(
            unsafe {
                sys::cuMemGetAllocationGranularity(
                    &mut granularity,
                    &prop,
                    sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            },
            "cuMemGetAllocationGranularity",
        )?;
        let size = size.div_ceil(granularity.max(1)) * granularity.max(1);

        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        curesult(
            unsafe { sys::cuMemCreate(&mut handle, size, &prop, 0) },
            "cuMemCreate",
        )?;
        let mut va: sys::CUdeviceptr = 0;
        if let Err(e) = curesult(
            unsafe { sys::cuMemAddressReserve(&mut va, size, granularity.max(1), 0, 0) },
            "cuMemAddressReserve",
        ) {
            unsafe { sys::cuMemRelease(handle) };
            return Err(e);
        }
        if let Err(e) = curesult(unsafe { sys::cuMemMap(va, size, 0, handle, 0) }, "cuMemMap") {
            unsafe {
                sys::cuMemAddressFree(va, size);
                sys::cuMemRelease(handle);
            }
            return Err(e);
        }
        let mut desc: sys::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        desc.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.__bindgen_anon_1.id = device_id as core::ffi::c_int;
        desc.flags = sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if let Err(e) = curesult(
            unsafe { sys::cuMemSetAccess(va, size, &desc, 1) },
            "cuMemSetAccess",
        ) {
            unsafe {
                sys::cuMemUnmap(va, size);
                sys::cuMemAddressFree(va, size);
                sys::cuMemRelease(handle);
            }
            return Err(e);
        }

        tracing::info!(
            device_id,
            size,
            granularity,
            va = format!("{va:#x}"),
            "Using VMM devbar region as tier-2 storage (device-alias, collision-free)"
        );

        Ok(Self {
            // SAFETY: a successful cuMemAddressReserve for a non-zero size
            // never returns a null VA.
            ptr: NonNull::new(va as *mut u8).expect("cuMemAddressReserve returned null"),
            len: size,
            device_id,
            backing: DevbarBacking::Vmm,
            handles: RegistrationHandles::new(),
            pinned: None,
            vmm: Some(VmmOwn {
                ctx,
                handle,
                va,
                size,
            }),
            io: None,
        })
    }

    /// True when this storage is backed by a BAR-mapped, mock, or VMM
    /// device-alias devbar region (everything except the pinned fallback).
    pub fn is_devbar_backed(&self) -> bool {
        self.backing != DevbarBacking::Pinned
    }
}

impl Drop for DevbarStorage {
    fn drop(&mut self) {
        self.handles.release();
        // The IOMEMORY registration must be released before the window's
        // munmap (the mapping must stay valid); bind the owning context
        // first (the Vmm Drop's bind_to_thread lesson).
        if let Some(io) = self.io.take() {
            if let Err(e) = io.ctx.bind_to_thread() {
                tracing::error!(
                    error = %e,
                    "devbar drop: bind_to_thread failed; leaking the IOMEMORY registration"
                );
            } else {
                let r = unsafe { sys::cuMemHostUnregister(self.ptr.as_ptr() as *mut libc::c_void) };
                if r != sys::CUresult::CUDA_SUCCESS {
                    tracing::error!(result = ?r, "devbar drop: cuMemHostUnregister failed");
                }
            }
        }
        if let Some(vmm) = self.vmm.take() {
            if let Err(e) = vmm.ctx.bind_to_thread() {
                tracing::error!(error = %e, "vmm drop: bind_to_thread failed; leaking VMM region");
                return;
            }
            let r = unsafe { sys::cuMemUnmap(vmm.va, vmm.size) };
            let r_addr = unsafe { sys::cuMemAddressFree(vmm.va, vmm.size) };
            let r_rel = unsafe { sys::cuMemRelease(vmm.handle) };
            tracing::debug!(result = ?r_addr, "vmm drop: cuMemAddressFree");
            tracing::debug!(result = ?r_rel, "vmm drop: cuMemRelease");
            if r != sys::CUresult::CUDA_SUCCESS {
                tracing::error!(result = ?r, "vmm drop: cuMemUnmap failed");
            }
        } else if self.pinned.is_none() {
            unsafe {
                libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len);
            }
        }
        // pinned Drop frees the pinned allocation
    }
}

impl Storage for DevbarStorage {
    fn storage_type(&self) -> StorageType {
        match self.backing {
            // Devbar (BAR-mapped / mock / VMM) pools stay Device on the
            // layout wire: no new serialized StorageType variant, and the
            // TypeId-keyed CUDA transfer arms keep routing devbar D2H/H2D.
            // For Vmm the Device stamp is additionally literally true (the
            // memory is device-resident at a real device VA). The registered
            // NIXL memory type comes from NixlDescriptor::mem_type (Bar
            // follows DYN_KVBM_DEVBAR_MEM_TYPE; Vram for Vmm, Dram for
            // mock/pinned).
            DevbarBacking::Bar | DevbarBacking::Mock | DevbarBacking::Vmm => {
                StorageType::Device(self.device_id)
            }
            // Pinned fallback is byte-for-byte identical to the existing
            // PinnedStorage host tier: StorageType::Pinned keeps
            // nixl_mem_type() => MemType::Dram consistent with the pool-level
            // Dram registration.
            DevbarBacking::Pinned => StorageType::Pinned,
        }
    }

    fn addr(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn size(&self) -> usize {
        self.len
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl StorageMemset for DevbarStorage {
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<(), StorageError> {
        let Some(end) = offset.checked_add(size) else {
            return Err(StorageError::OperationFailed(
                "memset: offset + size overflows usize".into(),
            ));
        };
        if end > self.len {
            return Err(StorageError::OperationFailed(
                "memset: offset + size > storage size".into(),
            ));
        }
        match self.backing {
            DevbarBacking::Vmm => {
                let Some(vmm) = self.vmm.as_ref() else {
                    return Err(StorageError::OperationFailed(
                        "vmm backing without state".into(),
                    ));
                };
                vmm.ctx
                    .bind_to_thread()
                    .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
                let dst = (self.ptr.as_ptr() as u64) + offset as u64;
                let r = unsafe { sys::cuMemsetD8_v2(dst, value, size) };
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(StorageError::OperationFailed(format!(
                        "cuMemsetD8_v2 failed: {r:?}"
                    )));
                }
            }
            DevbarBacking::Bar | DevbarBacking::Mock | DevbarBacking::Pinned => unsafe {
                let ptr = self.ptr.as_ptr().add(offset);
                std::ptr::write_bytes(ptr, value, size);
            },
        }
        Ok(())
    }
}

impl RegisterableStorage for DevbarStorage {
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        self.handles.register(key, handle)
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

// DevbarStorage — NIXL support

impl NixlAccessible for DevbarStorage {}
impl NixlRegisterableStorage for DevbarStorage {}

impl MemoryRegion for DevbarStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    fn size(&self) -> usize {
        self.len
    }
}

impl NixlDescriptor for DevbarStorage {
    fn mem_type(&self) -> MemType {
        match self.backing {
            // The Vmm region IS device memory at a real device VA — Vram is
            // semantically correct on every system class; no per-class probe.
            DevbarBacking::Vmm => MemType::Vram,
            // The BAR-mapped region: the NIXL mem type follows
            // DYN_KVBM_DEVBAR_MEM_TYPE — auto/dram (default; the
            // 2026-10-04 evidence — a Vram registration of the raw sysfs
            // mmap fails at ucp_mem_map with EIO on PCIe) or vram for
            // peer-mapped products.
            DevbarBacking::Bar => {
                match std::env::var(env_devbar::DYN_KVBM_DEVBAR_MEM_TYPE)
                    .as_deref()
                    .unwrap_or("auto")
                {
                    "vram" => MemType::Vram,
                    _ => MemType::Dram,
                }
            }
            // The mock region is an anonymous host mapping, not device
            // memory: a Vram registration of host memory fails in UCX's
            // cuda transport on real-GPU hosts ("Backend error occurred").
            DevbarBacking::Mock => MemType::Dram,
            // Pinned fallback behaves like the existing host tier.
            DevbarBacking::Pinned => MemType::Dram,
        }
    }

    fn device_id(&self) -> u64 {
        self.device_id as u64
    }
}

/// Allocator for [`DevbarStorage`].
///
/// Selects the backing from the environment:
/// - `DYN_KVBM_DEVBAR_BDF=<bdf>` — mmap the BAR aperture of that PCI device;
///   the sysfs resource file is resolved at `allocate()` time
///   (smallest O_RDWR-openable `resource{N}` that fits the tier-2 size), or
///   `DYN_KVBM_DEVBAR_RESOURCE_INDEX=<n>` pins the index explicitly
///   (e.g. `0000:41:00.0` -> `/sys/bus/pci/devices/0000:41:00.0/resourceN`)
/// - `DYN_KVBM_DEVBAR_MOCK=1` — anonymous mapping (plumbing tests without hardware)
/// - `DYN_KVBM_DEVBAR_VMM=1` — CUDA-VMM device-alias allocation (cuMemCreate +
///   cuMemMap at a reserved GPU VA); mock wins over VMM, VMM wins over BDF
/// - otherwise — standard pinned host allocation (default deployments unaffected)
///
/// The tier-2 sizing still comes from `DYN_KVBM_CPU_CACHE_GB` /
/// `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS` (computed upstream into num_host_blocks);
/// allocation fails if the requested size exceeds the devbar region.
#[derive(Debug, Default)]
pub struct DevbarAllocator {
    region: Option<DevbarRegion>,
    device_id: u32,
    /// The io mapping stashed by the last Bar/Bdf allocate (present only when
    /// the IOMEMORY registration succeeded): (host base, device base, length).
    /// The worker holds the allocator, not the storage — this is the channel.
    /// Per-group tier pools read it after EACH allocate (each window has its
    /// own mapping).
    io_mapping: std::sync::Mutex<Option<(usize, u64, usize)>>,
    /// Byte offset (from the configured region offset) already carved out of
    /// the aperture by previous Bar/Bdf allocates. Sequential allocates must
    /// get DISJOINT windows — per-group tier pools allocate one window per
    /// group, and two windows at the same aperture offset would alias the
    /// same physical BAR bytes. Mock/Vmm/Pinned allocations are independent
    /// mappings and never touch this cursor.
    aperture_cursor: std::sync::Mutex<u64>,
}

#[derive(Debug, Clone)]
enum DevbarRegion {
    /// Explicit BAR aperture file (constructor / tests).
    Bar {
        path: PathBuf,
        offset: u64,
    },
    /// BDF-selected aperture: the sysfs resource file is resolved at
    /// `allocate()` time via `select_resource_file` (needs the tier-2
    /// size, which only exists there). `explicit_index` comes from
    /// `DYN_KVBM_DEVBAR_RESOURCE_INDEX`.
    Bdf {
        bdf: String,
        offset: u64,
        explicit_index: Option<u32>,
    },
    Mock,
    Vmm,
}

/// Pick the BAR aperture file for `bdf`: the explicit index when set, else
/// the smallest O_RDWR-openable `resource{N}` whose size >= `tier2_size`
/// (the 16-MiB control BAR never fits a real tier-2). `sysfs_root` enables
/// tests; production callers pass "/sys/bus/pci/devices".
fn select_resource_file(
    bdf: &str,
    tier2_size: usize,
    explicit_index: Option<u32>,
    sysfs_root: &Path,
) -> Result<PathBuf, StorageError> {
    if let Some(idx) = explicit_index {
        let p = sysfs_root.join(bdf).join(format!("resource{idx}"));
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&p)
            .map_err(|e| {
                StorageError::InvalidConfig(format!("cannot open {}: {e}", p.display()))
            })?;
        let len = f
            .metadata()
            .map_err(|e| StorageError::InvalidConfig(e.to_string()))?
            .len();
        if (len as usize) < tier2_size {
            return Err(StorageError::InvalidConfig(format!(
                "{} is {len} bytes < tier-2 {tier2_size}; increase DYN_KVBM_DEVBAR_RESOURCE_INDEX or unset it to auto-select",
                p.display()
            )));
        }
        return Ok(p);
    }
    let mut candidates: Vec<(u64, PathBuf)> = Vec::new();
    // sysfs exposes resource0..resource15 (BARs + ROM + bridge windows); 16 covers every possible index.
    for idx in 0..16 {
        let p = sysfs_root.join(bdf).join(format!("resource{idx}"));
        let Ok(f) = std::fs::OpenOptions::new().read(true).write(true).open(&p) else {
            continue;
        };
        let Ok(meta) = f.metadata() else { continue };
        if meta.len() as usize >= tier2_size {
            candidates.push((meta.len(), p));
        }
    }
    candidates
        .into_iter()
        .min_by_key(|(len, _)| *len)
        .map(|(_, p)| p)
        .ok_or_else(|| {
            StorageError::InvalidConfig(format!(
                "no O_RDWR resource{{N}} >= {tier2_size} bytes under {}/{}; check DYN_KVBM_DEVBAR_BDF, or pin DYN_KVBM_DEVBAR_RESOURCE_INDEX",
                sysfs_root.display(),
                bdf
            ))
        })
}

impl DevbarAllocator {
    /// Build an allocator from the `DYN_KVBM_DEVBAR_*` environment variables.
    pub fn from_env() -> Result<Self, StorageError> {
        let device_id = std::env::var(env_devbar::DYN_KVBM_DEVBAR_DEVICE_ID)
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);

        if std::env::var(env_devbar::DYN_KVBM_DEVBAR_MOCK).is_ok_and(|v| v == "1") {
            tracing::info!("Using mock devbar region (anonymous mapping) as tier-2 storage");
            return Ok(Self {
                region: Some(DevbarRegion::Mock),
                device_id,
                io_mapping: std::sync::Mutex::new(None),
                aperture_cursor: std::sync::Mutex::new(0),
            });
        }

        if std::env::var(env_devbar::DYN_KVBM_DEVBAR_VMM).is_ok_and(|v| v == "1") {
            tracing::info!(
                device_id,
                "Using VMM devbar region (CUDA-managed, collision-free) as tier-2 storage"
            );
            return Ok(Self {
                region: Some(DevbarRegion::Vmm),
                device_id,
                io_mapping: std::sync::Mutex::new(None),
                aperture_cursor: std::sync::Mutex::new(0),
            });
        }

        if let Ok(bdf) = std::env::var(env_devbar::DYN_KVBM_DEVBAR_BDF) {
            let offset = std::env::var(env_devbar::DYN_KVBM_DEVBAR_OFFSET)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            let explicit_index = std::env::var(env_devbar::DYN_KVBM_DEVBAR_RESOURCE_INDEX)
                .ok()
                .and_then(|v| match v.parse::<u32>() {
                    Ok(idx) => Some(idx),
                    Err(_) => {
                        tracing::warn!(
                            value = %v,
                            "malformed DYN_KVBM_DEVBAR_RESOURCE_INDEX; ignoring (auto-select)"
                        );
                        None
                    }
                });
            // No sysfs access here: the aperture file is resolved at
            // allocate() time, when the tier-2 size is known (from_env must
            // succeed on machines without the device).
            tracing::info!(
                bdf,
                offset,
                explicit_index = ?explicit_index,
                device_id,
                "Using devbar region as tier-2 storage (aperture resolved at allocate time)"
            );
            return Ok(Self {
                region: Some(DevbarRegion::Bdf {
                    bdf,
                    offset,
                    explicit_index,
                }),
                device_id,
                io_mapping: std::sync::Mutex::new(None),
                aperture_cursor: std::sync::Mutex::new(0),
            });
        }

        Ok(Self {
            region: None,
            device_id,
            io_mapping: std::sync::Mutex::new(None),
            aperture_cursor: std::sync::Mutex::new(0),
        })
    }

    /// Allocator over an anonymous mock mapping (tests).
    pub fn mock(device_id: u32) -> Self {
        Self {
            region: Some(DevbarRegion::Mock),
            device_id,
            io_mapping: std::sync::Mutex::new(None),
            aperture_cursor: std::sync::Mutex::new(0),
        }
    }

    /// Allocator over an explicit BAR aperture (tests / programmatic use).
    pub fn bar(path: PathBuf, offset: u64, device_id: u32) -> Self {
        Self {
            region: Some(DevbarRegion::Bar { path, offset }),
            device_id,
            io_mapping: std::sync::Mutex::new(None),
            aperture_cursor: std::sync::Mutex::new(0),
        }
    }

    /// Allocator over a CUDA-VMM device-alias region (tests / programmatic use).
    pub fn vmm(device_id: u32) -> Self {
        Self {
            region: Some(DevbarRegion::Vmm),
            device_id,
            io_mapping: std::sync::Mutex::new(None),
            aperture_cursor: std::sync::Mutex::new(0),
        }
    }

    /// The [`StorageType`] that blocks of this allocator's backing carry:
    /// `Device(device_id)` for a BAR-mapped, mock, or VMM device-alias devbar
    /// region, `Pinned` for the standard pinned fallback (identical to
    /// today's host tier).
    /// Matches [`DevbarStorage::storage_type`] for every backing this
    /// allocator can produce; host-tier block factories stamp this.
    /// Exception to block-level/pool-level mem-type consistency: mock blocks
    /// stamp `Device` (wire semantics) while the pool-level registration is
    /// deliberately `Dram` (UCX rejects a Vram registration of host memory);
    /// the Bar backing does the same under its default (`Dram` pool-level
    /// registration via `DYN_KVBM_DEVBAR_MEM_TYPE`); only the pinned fallback
    /// is consistent at both levels.
    pub fn host_storage_type(&self) -> StorageType {
        match &self.region {
            Some(_) => StorageType::Device(self.device_id),
            None => StorageType::Pinned,
        }
    }

    /// True when this allocator's region is a VMM device-alias — the tier-2
    /// memory is device-resident and its transfers must use the D2D strategy
    /// (see `TransferContext::host_transfer_adjust`).
    pub fn is_device_aliased(&self) -> bool {
        matches!(self.region, Some(DevbarRegion::Vmm))
    }

    /// The transfer semantics this allocator's backing requires, honoring
    /// the `DYN_KVBM_DEVBAR_TRANSFERS` override. Evidence-based default for
    /// Bar on a raw sysfs aperture: staged (a failed direct copy is a sticky
    /// context poison — Xid 31, 2026-10-06); `direct` is opt-in for
    /// peer-mapped products whose acceptance kit passed.
    pub fn transfer_semantics(
        &self,
    ) -> crate::block_manager::block::transfer::HostTransferSemantics {
        use crate::block_manager::block::transfer::HostTransferSemantics as S;
        match &self.region {
            Some(DevbarRegion::Vmm) => S::DeviceAlias,
            Some(DevbarRegion::Bar { .. }) | Some(DevbarRegion::Bdf { .. }) => {
                match std::env::var(env_devbar::DYN_KVBM_DEVBAR_TRANSFERS)
                    .as_deref()
                    .unwrap_or("auto")
                {
                    "auto" | "staged" => S::StagedCpu,
                    "direct" => S::DirectIo,
                    other => {
                        tracing::warn!(
                            value = other,
                            "unrecognized DYN_KVBM_DEVBAR_TRANSFERS; using staged (default)"
                        );
                        S::StagedCpu
                    }
                }
            }
            _ => S::HostPointer,
        }
    }

    /// The io mapping stashed by the last Bar/Bdf allocate (present only when
    /// the IOMEMORY registration succeeded): (host base, device base, length).
    pub fn io_mapping(&self) -> Option<(usize, u64, usize)> {
        *self.io_mapping.lock().unwrap()
    }
}

impl StorageAllocator<DevbarStorage> for DevbarAllocator {
    fn allocate(&self, size: usize) -> Result<DevbarStorage, StorageError> {
        // Bar/Bdf carve sequential, page-rounded windows out of one aperture
        // (two windows at the same offset would alias the same physical BAR
        // bytes). The carve is rolled back if the mapping fails so a failed
        // allocate does not permanently consume the aperture.
        let carve = match &self.region {
            Some(DevbarRegion::Bar { offset, .. }) | Some(DevbarRegion::Bdf { offset, .. }) => {
                Some(self.reserve_aperture(*offset, size))
            }
            _ => None,
        };
        let mapped = match (&self.region, carve) {
            (Some(DevbarRegion::Bar { path, .. }), Some((carve_offset, _))) => {
                DevbarStorage::from_bar(path.clone(), carve_offset, size, self.device_id)
            }
            (
                Some(DevbarRegion::Bdf {
                    bdf,
                    explicit_index,
                    ..
                }),
                Some((carve_offset, _)),
            ) => {
                match select_resource_file(
                    bdf,
                    size,
                    *explicit_index,
                    Path::new("/sys/bus/pci/devices"),
                ) {
                    Ok(path) => DevbarStorage::from_bar(path, carve_offset, size, self.device_id),
                    Err(e) => Err(e),
                }
            }
            (Some(DevbarRegion::Mock), _) => DevbarStorage::mock(size, self.device_id),
            (Some(DevbarRegion::Vmm), _) => DevbarStorage::from_vmm(size, self.device_id),
            (None, _) => DevbarStorage::pinned_fallback(size, Some(self.device_id)),
            _ => unreachable!("carve is set exactly when the region is Bar/Bdf"),
        };
        let storage = match mapped {
            Ok(storage) => storage,
            Err(e) => {
                if let Some((_, consumed)) = carve {
                    self.rollback_aperture(consumed);
                }
                return Err(e);
            }
        };
        // Stash the io mapping for the worker's wiring (the worker holds the
        // allocator, not the storage). Set unconditionally: None clears a
        // stale mapping from a previous allocate (strictly more robust).
        // Per-group tier pools read it after EACH allocate — every window
        // carries its own mapping.
        *self.io_mapping.lock().unwrap() = storage.io_mapping();
        Ok(storage)
    }
}

impl DevbarAllocator {
    /// Reserve the next Bar/Bdf aperture window: carve offset (absolute mmap
    /// offset) plus the page-rounded consumption to roll back on failure.
    fn reserve_aperture(&self, base_offset: u64, size: usize) -> (u64, u64) {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        // sysconf cannot fail for _SC_PAGESIZE on Linux; a 0 would mean a
        // broken libc — guard anyway so div_ceil never panics.
        let page = page.max(1);
        let consumed = (size as u64).div_ceil(page) * page;
        let mut cursor = self.aperture_cursor.lock().unwrap();
        let carve = base_offset + *cursor;
        *cursor += consumed;
        (carve, consumed)
    }

    /// Undo a reserve whose mapping failed.
    fn rollback_aperture(&self, consumed: u64) {
        *self.aperture_cursor.lock().unwrap() -= consumed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_devbar_mock_roundtrip_and_descriptor() {
        let allocator = DevbarAllocator::mock(3);
        let mut storage = allocator.allocate(4096).expect("mock devbar allocation");

        assert_eq!(Storage::size(&storage), 4096);
        assert_ne!(storage.addr(), 0, "Address should be non-zero");
        assert!(storage.is_devbar_backed());
        // The mock region is an anonymous HOST mapping: StorageType stays
        // Device (pool/wire semantics unchanged), but the NIXL descriptor
        // must report Dram — UCX's cuda transport rejects a Vram
        // registration of host memory on real-GPU hosts.
        assert_eq!(storage.storage_type(), StorageType::Device(3));
        // The descriptor triple required by the KVBM design doc:
        // address + size, device ID, memory type.
        assert_eq!(storage.device_id(), 3);
        assert_eq!(storage.mem_type(), MemType::Dram);

        unsafe {
            let ptr = storage.as_mut_ptr();
            for i in 0..4096 {
                std::ptr::write_volatile(ptr.add(i), (i & 0xFF) as u8);
            }
            for i in 0..4096 {
                assert_eq!(std::ptr::read_volatile(ptr.add(i)), (i & 0xFF) as u8);
            }
        }

        storage.memset(0xAB, 0, 4096).unwrap();
        unsafe {
            assert_eq!(std::ptr::read_volatile(Storage::as_ptr(&storage)), 0xAB);
        }
    }

    /// Requires CUDA (pinned allocation); fails with a cudarc libcuda load
    /// error on machines without a GPU or driver.
    #[test]
    fn test_devbar_pinned_fallback_descriptor() {
        let allocator = DevbarAllocator::default();
        let storage = allocator
            .allocate(8192)
            .expect("pinned fallback allocation");

        assert!(!storage.is_devbar_backed());
        assert_eq!(storage.storage_type(), StorageType::Pinned);
        assert_eq!(storage.device_id(), 0);
        assert_eq!(storage.mem_type(), MemType::Dram);
    }

    #[test]
    fn test_devbar_no_descriptor_before_registration() {
        let storage = DevbarStorage::mock(1024, 0).unwrap();
        assert!(unsafe { storage.as_nixl_descriptor() }.is_none());
    }

    #[test]
    fn test_devbar_from_bar_roundtrip_and_aperture_validation() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("DYN_KVBM_DEVBAR_MEM_TYPE") };
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        use std::io::{Seek as _, Write as _};
        file.write_all(&[0x5A; 1024]).expect("write pattern");
        file.flush().expect("flush");
        file.rewind().expect("rewind");

        let storage = DevbarStorage::from_bar(file.path().to_path_buf(), 0, 1024, 7)
            .expect("mmap of ordinary file as devbar region");
        assert!(storage.is_devbar_backed());
        assert_eq!(storage.storage_type(), StorageType::Device(7));
        assert_eq!(storage.device_id(), 7);
        // auto default (DYN_KVBM_DEVBAR_MEM_TYPE unset): Dram — the
        // 2026-10-04 evidence (a Vram registration of the raw sysfs mmap
        // fails at ucp_mem_map with EIO on PCIe).
        assert_eq!(storage.mem_type(), MemType::Dram);
        unsafe {
            assert_eq!(std::ptr::read_volatile(Storage::as_ptr(&storage)), 0x5A);
        }
        drop(storage);

        // vram override: a second from_bar on the same path at offset 0
        // (the first storage released its mmap above).
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_MEM_TYPE", "vram") };
        let storage_vram = DevbarStorage::from_bar(file.path().to_path_buf(), 0, 1024, 7)
            .expect("mmap with vram mem-type override");
        assert_eq!(storage_vram.mem_type(), MemType::Vram);
        drop(storage_vram);
        unsafe { std::env::remove_var("DYN_KVBM_DEVBAR_MEM_TYPE") };

        // Requesting past the aperture must be rejected with the sizing
        // guidance error.
        let err = DevbarStorage::from_bar(file.path().to_path_buf(), 0, 2048, 7)
            .expect_err("aperture overflow must be rejected");
        assert!(matches!(err, StorageError::InvalidConfig(_)));

        // Unaligned offsets must be rejected with a clear error.
        let err = DevbarStorage::from_bar(file.path().to_path_buf(), 1, 512, 7)
            .expect_err("unaligned offset must be rejected");
        assert!(matches!(err, StorageError::InvalidConfig(_)));
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// `DYN_KVBM_DEVBAR_VMM=1` selects the Vmm region; mock wins over VMM;
    /// VMM alone (no BDF) selects Vmm; nothing set = pinned fallback.
    #[test]
    fn from_env_selects_vmm_region() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: devbar env vars are only read by devbar code and this test;
        // ENV_LOCK guards against parallel env mutation within this module.
        unsafe { std::env::remove_var("DYN_KVBM_DEVBAR_BDF") };
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_VMM", "1") };
        let alloc = DevbarAllocator::from_env().unwrap();
        assert!(matches!(alloc.region, Some(DevbarRegion::Vmm)));
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_MOCK", "1") };
        let alloc = DevbarAllocator::from_env().unwrap();
        assert!(matches!(alloc.region, Some(DevbarRegion::Mock)));
        unsafe { std::env::remove_var("DYN_KVBM_DEVBAR_MOCK") };
        let alloc = DevbarAllocator::from_env().unwrap();
        assert!(matches!(alloc.region, Some(DevbarRegion::Vmm)));
        assert!(alloc.is_device_aliased());
        unsafe { std::env::remove_var("DYN_KVBM_DEVBAR_VMM") };
        let alloc = DevbarAllocator::from_env().unwrap();
        assert!(alloc.region.is_none());
        assert!(!alloc.is_device_aliased());
    }

    /// BAR-backed transfer semantics + resource-index plumbing. Works on
    /// machines without the device: from_env does no sysfs access (the
    /// aperture is resolved at allocate() time).
    #[test]
    fn from_env_bar_transfer_semantics_and_resource_index() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("DYN_KVBM_DEVBAR_TRANSFERS") };
        unsafe { std::env::remove_var("DYN_KVBM_DEVBAR_MEM_TYPE") };
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_BDF", "0000:41:00.0") };
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_RESOURCE_INDEX", "0") };
        let alloc = DevbarAllocator::from_env().unwrap();
        assert!(matches!(
            alloc.transfer_semantics(),
            crate::block_manager::block::transfer::HostTransferSemantics::StagedCpu
        ));
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_TRANSFERS", "direct") };
        assert!(matches!(
            alloc.transfer_semantics(),
            crate::block_manager::block::transfer::HostTransferSemantics::DirectIo
        ));
        unsafe {
            std::env::remove_var("DYN_KVBM_DEVBAR_TRANSFERS");
            std::env::remove_var("DYN_KVBM_DEVBAR_RESOURCE_INDEX");
            std::env::remove_var("DYN_KVBM_DEVBAR_BDF");
        }
    }

    /// `TRANSFERS=direct` on a BAR-backed allocator maps to the DirectIo
    /// semantics (the IOMEMORY-registered device-VA path from Task 2).
    #[test]
    fn from_env_bar_direct_maps_to_direct_io_semantics() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_BDF", "0000:41:00.0") };
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_RESOURCE_INDEX", "0") };
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_TRANSFERS", "direct") };
        let alloc = DevbarAllocator::from_env().unwrap();
        assert!(matches!(
            alloc.transfer_semantics(),
            crate::block_manager::block::transfer::HostTransferSemantics::DirectIo
        ));
        unsafe {
            std::env::remove_var("DYN_KVBM_DEVBAR_TRANSFERS");
            std::env::remove_var("DYN_KVBM_DEVBAR_RESOURCE_INDEX");
            std::env::remove_var("DYN_KVBM_DEVBAR_BDF");
        }
    }

    #[test]
    fn select_resource_file_prefers_smallest_fitting() {
        let dir = std::env::temp_dir().join(format!("devbar-sysfs-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("0000:41:00.0")).unwrap();
        for (idx, sz) in [(0, 16u64 << 20), (1, 64u64 << 30), (2, 2u64 << 30)] {
            let p = dir.join("0000:41:00.0").join(format!("resource{idx}"));
            std::fs::write(&p, vec![0u8; 8]).unwrap(); // placeholder content
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&p)
                .unwrap();
            f.set_len(sz).unwrap();
            drop(f);
        }
        // 1-GiB tier-2: 16 MiB (resource0) does not fit; smallest fitting = 2 GiB (resource2).
        let picked = select_resource_file("0000:41:00.0", 1 << 30, None, &dir).unwrap();
        assert!(picked.ends_with("resource2"));
        // Explicit index wins when it fits.
        let explicit = select_resource_file("0000:41:00.0", 1 << 30, Some(1), &dir).unwrap();
        assert!(explicit.ends_with("resource1"));
        let explicit2 = select_resource_file("0000:41:00.0", 1 << 30, Some(2), &dir).unwrap();
        assert!(explicit2.ends_with("resource2"));
        // Explicit index that does not fit → error.
        assert!(select_resource_file("0000:41:00.0", 1 << 30, Some(0), &dir).is_err());
        // Nothing large enough → error.
        assert!(select_resource_file("0000:41:00.0", 128 << 30, None, &dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Direct-io roundtrip through the REAL BAR: H2D and D2H via the
    /// IOMEMORY-registered device VA (the executor's shape). REQUIRES root
    /// (the IOMEMORY registration's privilege check) and the expander
    /// stand-in (the BAR aperture as resource0); run on the node with the
    /// `testing-cuda` feature. A non-root run fails in `DevbarStorage::from_bar`
    /// at the registration with CUDA_ERROR_NOT_PERMITTED (the documented
    /// privilege requirement); a no-GPU machine fails earlier at cudarc /
    /// context setup.
    #[cfg(feature = "testing-cuda")]
    #[test]
    fn test_devbar_direct_io_roundtrip_real_bar() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_BDF", "0000:41:00.0") };
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_RESOURCE_INDEX", "0") };
        unsafe { std::env::set_var("DYN_KVBM_DEVBAR_TRANSFERS", "direct") };
        let alloc = DevbarAllocator::from_env().unwrap();
        assert!(matches!(
            alloc.transfer_semantics(),
            crate::block_manager::block::transfer::HostTransferSemantics::DirectIo
        ));
        let mut storage = alloc
            .allocate(64 * 1024)
            .expect("direct alloc (root + the expander stand-in needed)");
        let (host_base, io_dev_base, len) = alloc.io_mapping().expect("io mapping wired");
        assert_eq!(len, 64 * 1024);

        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        ctx.bind_to_thread().unwrap();

        // H2D: stamp a pattern into the window via the host VA (CPU),
        // translate to the device VA, memcpy to a device buffer, DtoH back,
        // compare — the executor's shape.
        let pattern: Vec<u8> = (0..64).map(|i| (0xA5 ^ i) as u8).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), storage.as_mut_ptr(), pattern.len());
        }
        let s_dev = io_dev_base + (storage.addr() as usize - host_base) as u64;
        let mut dev: sys::CUdeviceptr = 0;
        unsafe {
            assert_eq!(
                sys::cuMemAlloc_v2(&mut dev, 64 * 1024),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                sys::cuMemcpyAsync(dev, s_dev, pattern.len(), std::ptr::null_mut()),
                sys::CUresult::CUDA_SUCCESS
            );
            let mut back = [0u8; 64];
            assert_eq!(
                sys::cuMemcpyDtoH_v2(back.as_mut_ptr() as *mut core::ffi::c_void, dev, 64),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                &back,
                pattern.as_slice(),
                "direct H2D through the io mapping"
            );
            assert_eq!(sys::cuMemFree_v2(dev), sys::CUresult::CUDA_SUCCESS);
        }

        // D2H: stamp a distinct pattern into VRAM, memcpy through the device
        // VA into the window, read back via the host VA (CPU), compare.
        let changed: Vec<u8> = (0..64).map(|i| (0x5C ^ i) as u8).collect();
        let mut dev2: sys::CUdeviceptr = 0;
        unsafe {
            assert_eq!(
                sys::cuMemAlloc_v2(&mut dev2, 64 * 1024),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                sys::cuMemcpyHtoD_v2(
                    dev2,
                    changed.as_ptr() as *const core::ffi::c_void,
                    changed.len()
                ),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                sys::cuMemcpyAsync(s_dev, dev2, changed.len(), std::ptr::null_mut()),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(sys::cuMemFree_v2(dev2), sys::CUresult::CUDA_SUCCESS);
        }
        let mut window_back = [0u8; 64];
        unsafe {
            std::ptr::copy_nonoverlapping(Storage::as_ptr(&storage), window_back.as_mut_ptr(), 64);
        }
        assert_eq!(
            window_back.as_slice(),
            changed.as_slice(),
            "direct D2H through the io mapping"
        );

        unsafe {
            std::env::remove_var("DYN_KVBM_DEVBAR_TRANSFERS");
            std::env::remove_var("DYN_KVBM_DEVBAR_RESOURCE_INDEX");
            std::env::remove_var("DYN_KVBM_DEVBAR_BDF");
        }
    }
}

#[cfg(all(test, feature = "testing-nixl"))]
mod nixl_tests {
    use super::super::nixl::NixlAgent;
    use super::*;

    /// Runs only where NIXL + the UCX plugin are installed. Uses the pinned
    /// fallback (Dram) because Vram registration requires a CUDA context.
    #[test]
    fn test_devbar_pinned_fallback_nixl_registration() {
        let agent = NixlAgent::new("devbar_test_agent").unwrap();
        let (_, params) = agent
            .get_plugin_params("UCX")
            .expect("UCX plugin required for this test");
        let _backend = agent.create_backend("UCX", &params).unwrap();

        let mut storage = DevbarStorage::pinned_fallback(4096, None).unwrap();
        assert!(unsafe { storage.as_nixl_descriptor() }.is_none());

        storage.nixl_register(&agent, None).unwrap();
        assert!(storage.is_nixl_registered());

        let desc = unsafe { storage.as_nixl_descriptor() }.expect("descriptor after registration");
        assert_eq!(*desc.size(), 4096);
        assert_eq!(*desc.mem_type(), MemType::Dram);
        assert_eq!(*desc.device_id(), 0);
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod vmm_tests {
    use super::*;
    use cudarc::driver::sys;

    const PATTERN: &[u8; 16] = b"VMM-UNIT-PATTERN";

    /// Vmm device-alias: descriptor triple, device memset, D2D both
    /// directions, leak-detected teardown.
    #[test]
    fn test_devbar_vmm_roundtrip_and_descriptor() {
        let alloc = DevbarAllocator::vmm(0);
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        ctx.bind_to_thread().unwrap();

        // Baseline free memory: a leaked cuMemCreate physical handle shows
        // up as free memory NOT returning to this baseline after teardown.
        let (free_before, _) = ctx.mem_get_info().unwrap();

        let mut storage = alloc.allocate(8 * 1024 * 1024).expect("from_vmm");

        // Descriptor triple: Device-stamped, Vram-typed, real device VA.
        assert_eq!(storage.storage_type(), StorageType::Device(0));
        assert_eq!(storage.mem_type(), MemType::Vram);
        assert_eq!(storage.device_id(), 0);
        assert!(storage.is_devbar_backed());
        let va = storage.addr();
        assert_ne!(va, 0);

        // Device-side memset via the Storage API (the Vmm arm).
        storage.memset(0xCD, 16, 32).unwrap();
        let mut probe = [0u8; 16];
        unsafe {
            assert_eq!(
                sys::cuMemcpyDtoH_v2(probe.as_mut_ptr() as *mut core::ffi::c_void, va + 16, 16),
                sys::CUresult::CUDA_SUCCESS
            );
        }
        assert_eq!(probe, [0xCD; 16]);

        // D2D both directions: alias <-> a normal device buffer.
        let mut dev: sys::CUdeviceptr = 0;
        unsafe {
            assert_eq!(
                sys::cuMemAlloc_v2(&mut dev, 16),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                sys::cuMemcpyHtoD_v2(va, PATTERN.as_ptr() as *const core::ffi::c_void, 16),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                sys::cuMemcpyDtoD_v2(dev, va, 16),
                sys::CUresult::CUDA_SUCCESS
            );
            let mut mid = [0u8; 16];
            assert_eq!(
                sys::cuMemcpyDtoH_v2(mid.as_mut_ptr() as *mut core::ffi::c_void, dev, 16),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(&mid, PATTERN);
            assert_eq!(
                sys::cuMemcpyDtoD_v2(va, dev, 16),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(sys::cuMemFree_v2(dev), sys::CUresult::CUDA_SUCCESS);
            let mut back = [0u8; 16];
            assert_eq!(
                sys::cuMemcpyDtoH_v2(back.as_mut_ptr() as *mut core::ffi::c_void, va, 16),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(&back, PATTERN);
        }

        // Drop: unmap + address-free + release must not panic or leak; the
        // physical handle's release shows up as free memory returning to the
        // baseline (2 MiB tolerance for accounting jitter). The re-allocation
        // below additionally proves re-allocation works after release.
        drop(storage);
        let (free_after, _) = ctx.mem_get_info().unwrap();
        assert!(
            free_after + (2 << 20) >= free_before,
            "VMM region leaked: free went {free_before} -> {free_after}"
        );
        let again = alloc
            .allocate(8 * 1024 * 1024)
            .expect("re-allocate after drop");
        drop(again);
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod staged_tests {
    use super::*;
    use cudarc::driver::sys;

    /// Staged legs' mechanics with real CUDA: the pattern goes window (CPU)
    /// -> bounce (CPU) -> device (DMA) -> bounce (DMA) -> window (CPU). The
    /// mock region is host RAM, so every leg is exercised without MMIO; the
    /// full executor + routing are proven by the e2e on real hardware.
    #[test]
    fn test_devbar_staged_roundtrip_mock() {
        let alloc = DevbarAllocator::mock(0);
        let mut storage = alloc.allocate(64 * 1024).expect("mock alloc");

        let pattern: &[u8; 15] = b"STAGED-PATTERN2";
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), storage.as_mut_ptr(), pattern.len());
        }

        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        ctx.bind_to_thread().unwrap();
        let mut dev: sys::CUdeviceptr = 0;
        unsafe {
            assert_eq!(
                sys::cuMemAlloc_v2(&mut dev, 64 * 1024),
                sys::CUresult::CUDA_SUCCESS
            );
            // H2D staged legs: CPU window->bounce, DMA bounce->device.
            let mut bounce = vec![0u8; 64 * 1024];
            std::ptr::copy_nonoverlapping(
                Storage::as_ptr(&storage),
                bounce.as_mut_ptr(),
                pattern.len(),
            );
            assert_eq!(
                sys::cuMemcpyHtoD_v2(
                    dev,
                    bounce.as_ptr() as *const core::ffi::c_void,
                    pattern.len()
                ),
                sys::CUresult::CUDA_SUCCESS
            );
            // D2H staged legs: DMA device->bounce, CPU bounce->window; compare.
            let mut back = [0u8; 15];
            assert_eq!(
                sys::cuMemcpyDtoH_v2(back.as_mut_ptr() as *mut core::ffi::c_void, dev, 15),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(&back, pattern, "device->bounce leg integrity");
            std::ptr::copy_nonoverlapping(back.as_ptr(), storage.as_mut_ptr(), back.len());
            let window = std::slice::from_raw_parts(Storage::as_ptr(&storage), 15);
            assert_eq!(window, pattern, "staged roundtrip pattern integrity");
            assert_eq!(sys::cuMemFree_v2(dev), sys::CUresult::CUDA_SUCCESS);
        }
        drop(storage);
    }
}

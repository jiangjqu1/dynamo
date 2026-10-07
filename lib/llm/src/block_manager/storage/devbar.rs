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
//! [`DevbarStorage`] implements [`NixlDescriptor`] returning `MemType::Vram`
//! with the configured device ID for BAR-backed instances (devbar is still GPU
//! memory), so the serialized descriptor carries the "Device ID + memory type"
//! triple required by the KVBM design doc's metadata-exchange section. The
//! mock region and the pinned fallback report `MemType::Dram` — both are host
//! memory, and a Vram registration of host memory fails in UCX's cuda
//! transport on real-GPU hosts. The Vmm backing reports `MemType::Vram` at a
//! real device VA — semantically correct on every system class (it *is*
//! device memory), so it needs no per-class probe.

use std::{path::PathBuf, ptr::NonNull};

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
/// (`/sys/bus/pci/devices/<bdf>/resource0` mmap), an anonymous mock mapping, or —
/// when no devbar region is configured — standard pinned allocation.
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
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
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

        Ok(Self {
            ptr,
            len: size,
            device_id,
            backing: DevbarBacking::Bar,
            handles: RegistrationHandles::new(),
            pinned: None,
            vmm: None,
        })
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
            // NIXL memory type comes from NixlDescriptor::mem_type (Vram for
            // Bar and Vmm, Dram for mock/pinned).
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
            // The BAR-mapped region is GPU memory: the serialized descriptor
            // carries the "Device ID + memory type" triple of the KVBM design
            // doc's metadata-exchange section. (PCIe caveat: UCX's cuda
            // transport cannot map the raw sysfs mmap as Vram on PCIe-only
            // GPUs — Bar mode is a C2C-targeted mechanism; the Vmm backing
            // is the PCIe answer.)
            DevbarBacking::Bar => MemType::Vram,
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
/// - `DYN_KVBM_DEVBAR_BDF=<bdf>` — mmap the BAR aperture of that PCI device
///   (e.g. `0000:1f:00.0` -> `/sys/bus/pci/devices/0000:1f:00.0/resource0`)
/// - `DYN_KVBM_DEVBAR_MOCK=1` — anonymous mapping (plumbing tests without hardware)
/// - `DYN_KVBM_DEVBAR_VMM=1` — CUDA-VMM device-alias allocation (cuMemCreate +
///   cuMemMap at a reserved GPU VA); mock wins over VMM, VMM wins over BDF
/// - otherwise — standard pinned host allocation (default deployments unaffected)
///
/// The tier-2 sizing still comes from `DYN_KVBM_CPU_CACHE_GB` /
/// `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS` (computed upstream into num_host_blocks);
/// allocation fails if the requested size exceeds the devbar region.
#[derive(Debug, Clone, Default)]
pub struct DevbarAllocator {
    region: Option<DevbarRegion>,
    device_id: u32,
}

#[derive(Debug, Clone)]
enum DevbarRegion {
    Bar { path: PathBuf, offset: u64 },
    Mock,
    Vmm,
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
            });
        }

        if let Ok(bdf) = std::env::var(env_devbar::DYN_KVBM_DEVBAR_BDF) {
            let path = PathBuf::from(format!("/sys/bus/pci/devices/{bdf}/resource0"));
            let offset = std::env::var(env_devbar::DYN_KVBM_DEVBAR_OFFSET)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            tracing::info!(
                bdf,
                path = %path.display(),
                offset,
                device_id,
                "Using devbar region as tier-2 storage"
            );
            return Ok(Self {
                region: Some(DevbarRegion::Bar { path, offset }),
                device_id,
            });
        }

        Ok(Self {
            region: None,
            device_id,
        })
    }

    /// Allocator over an anonymous mock mapping (tests).
    pub fn mock(device_id: u32) -> Self {
        Self {
            region: Some(DevbarRegion::Mock),
            device_id,
        }
    }

    /// Allocator over an explicit BAR aperture (tests / programmatic use).
    pub fn bar(path: PathBuf, offset: u64, device_id: u32) -> Self {
        Self {
            region: Some(DevbarRegion::Bar { path, offset }),
            device_id,
        }
    }

    /// Allocator over a CUDA-VMM device-alias region (tests / programmatic use).
    pub fn vmm(device_id: u32) -> Self {
        Self {
            region: Some(DevbarRegion::Vmm),
            device_id,
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
    /// Bar and pinned-fallback backings are consistent at both levels.
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
}

impl StorageAllocator<DevbarStorage> for DevbarAllocator {
    fn allocate(&self, size: usize) -> Result<DevbarStorage, StorageError> {
        match &self.region {
            Some(DevbarRegion::Bar { path, offset }) => {
                DevbarStorage::from_bar(path.clone(), *offset, size, self.device_id)
            }
            Some(DevbarRegion::Mock) => DevbarStorage::mock(size, self.device_id),
            Some(DevbarRegion::Vmm) => DevbarStorage::from_vmm(size, self.device_id),
            None => DevbarStorage::pinned_fallback(size, Some(self.device_id)),
        }
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
        assert_eq!(storage.mem_type(), MemType::Vram);
        unsafe {
            assert_eq!(std::ptr::read_volatile(Storage::as_ptr(&storage)), 0x5A);
        }
        drop(storage);

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
        let _guard = ENV_LOCK.lock().unwrap();
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

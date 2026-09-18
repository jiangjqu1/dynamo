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
//! pinned fallback reports `MemType::Dram`, matching the existing host tier.

use std::{path::PathBuf, ptr::NonNull};

use super::nixl::{MemType, MemoryRegion, NixlAccessible, NixlDescriptor, NixlRegisterableStorage};
use super::{
    CudaAccessible, Local, RegistationHandle, RegisterableStorage, RegistrationHandles, Storage,
    StorageAllocator, StorageError, StorageMemset, StorageType, SystemAccessible,
};
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
}

// SAFETY: DevbarStorage owns its mapping exclusively; raw-pointer access is
// confined to the `Storage`/`StorageMemset` unsafe APIs, mirroring SystemStorage.
unsafe impl Send for DevbarStorage {}
unsafe impl Sync for DevbarStorage {}

impl Local for DevbarStorage {}
impl SystemAccessible for DevbarStorage {}
impl CudaAccessible for DevbarStorage {}

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
                "devbar offset {offset} is not page-aligned; sysfs resource mmap requires \
                 a page-aligned offset of {}.",
                path.display()
            )));
        }
        let Some(request_end) = offset.checked_add(size as u64) else {
            unsafe { libc::close(fd) };
            return Err(StorageError::InvalidConfig(format!(
                "devbar request (offset {offset} + {size} bytes) overflows of {}.",
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
        })
    }

    /// True when this storage is backed by a BAR-mapped (or mock) devbar region.
    pub fn is_devbar_backed(&self) -> bool {
        self.backing != DevbarBacking::Pinned
    }
}

impl Drop for DevbarStorage {
    fn drop(&mut self) {
        self.handles.release();
        if self.pinned.is_none() {
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
            // Devbar (BAR-mapped / mock) is device memory: the Device variant
            // keeps nixl_mem_type() => MemType::Vram consistent with the
            // NixlDescriptor below, and avoids adding a new serialized
            // StorageType variant on the worker->peer layout wire.
            DevbarBacking::Bar | DevbarBacking::Mock => StorageType::Device(self.device_id),
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
        if offset + size > self.len {
            return Err(StorageError::OperationFailed(
                "memset: offset + size > storage size".into(),
            ));
        }
        unsafe {
            let ptr = self.ptr.as_ptr().add(offset);
            std::ptr::write_bytes(ptr, value, size);
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
            // Devbar (BAR-mapped / mock) is still GPU memory; the serialized
            // descriptor carries "Device ID + memory type" exactly as the KVBM
            // design doc's metadata-exchange section specifies.
            DevbarBacking::Bar | DevbarBacking::Mock => MemType::Vram,
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
/// - When both `DYN_KVBM_DEVBAR_MOCK=1` and `DYN_KVBM_DEVBAR_BDF` are set, the mock region wins.
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

    /// The [`StorageType`] that blocks of this allocator's backing carry:
    /// `Device(device_id)` for a BAR-mapped or mock devbar region (devbar is
    /// still GPU memory), `Pinned` for the standard pinned fallback (identical
    /// to today's host tier). Matches [`DevbarStorage::storage_type`] for
    /// every backing this allocator can produce; host-tier block factories
    /// stamp this so block-level NIXL mem types stay consistent with the
    /// pool-level registration.
    pub fn host_storage_type(&self) -> StorageType {
        match &self.region {
            Some(_) => StorageType::Device(self.device_id),
            None => StorageType::Pinned,
        }
    }
}

impl StorageAllocator<DevbarStorage> for DevbarAllocator {
    fn allocate(&self, size: usize) -> Result<DevbarStorage, StorageError> {
        match &self.region {
            Some(DevbarRegion::Bar { path, offset }) => {
                DevbarStorage::from_bar(path.clone(), *offset, size, self.device_id)
            }
            Some(DevbarRegion::Mock) => DevbarStorage::mock(size, self.device_id),
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
        // Devbar is device memory: the StorageType stays Device so
        // nixl_mem_type() => Vram stays consistent with NixlDescriptor.
        assert_eq!(storage.storage_type(), StorageType::Device(3));
        // The descriptor triple required by the KVBM design doc:
        // address + size, device ID, memory type.
        assert_eq!(storage.device_id(), 3);
        assert_eq!(storage.mem_type(), MemType::Vram);

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

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::TransferError;
use crate::block_manager::block::{BlockDataProvider, BlockDataProviderMut};

use anyhow::Result;
use cudarc::driver::CudaStream;
use cudarc::driver::result as cuda_result;
use cudarc::driver::sys::{CUevent_flags, CUresult, cuMemcpyHtoDAsync_v2};
use dynamo_runtime::config::environment_names::cuda as env_cuda;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::LazyLock;
use std::sync::Mutex;

// Global storage for kernel function - store as usize to avoid Send/Sync issues
static COPY_KERNEL_MODULE: Mutex<Option<usize>> = Mutex::new(None);
static COPY_KERNEL_FUNCTION: Mutex<Option<usize>> = Mutex::new(None);

type CudaMemcpyFnPtr = unsafe fn(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError>;

fn cuda_memcpy_fn_ptr(strategy: &TransferStrategy) -> Result<CudaMemcpyFnPtr, TransferError> {
    match strategy {
        TransferStrategy::CudaAsyncH2D => Ok(cuda_memcpy_h2d),
        TransferStrategy::CudaAsyncD2H => Ok(cuda_memcpy_d2h),
        TransferStrategy::CudaAsyncD2D => Ok(cuda_memcpy_d2d),
        _ => Err(TransferError::ExecutionError(
            "Unsupported copy strategy for CUDA memcpy async".into(),
        )),
    }
}

/// Collect K/V cache addresses from source and destination blocks
fn collect_kv_addresses<Source, Destination>(
    sources: &[Source],
    destinations: &[Destination],
    num_layers: usize,
    num_outer_dims: usize,
) -> Result<(Vec<u64>, Vec<u64>), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    if sources.is_empty() {
        return Err(TransferError::ExecutionError(
            "No source blocks provided".to_string(),
        ));
    }

    let total_address_pairs = sources.len() * num_layers * num_outer_dims;
    let mut src_addresses = Vec::with_capacity(total_address_pairs);
    let mut dst_addresses = Vec::with_capacity(total_address_pairs);

    let src_block_data: Vec<_> = sources.iter().map(|block| block.block_data()).collect();
    let dst_block_data: Vec<_> = destinations
        .iter()
        .map(|block| block.block_data())
        .collect();

    for (src_data, dst_data) in src_block_data.iter().zip(dst_block_data.iter()) {
        for layer_idx in 0..num_layers {
            for outer_idx in 0..num_outer_dims {
                let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                let dst_view = dst_data.layer_view(layer_idx, outer_idx)?;

                unsafe {
                    src_addresses.push(src_view.as_ptr() as u64);
                    dst_addresses.push(dst_view.as_ptr() as u64);
                }
            }
        }
    }

    Ok((src_addresses, dst_addresses))
}

/// The copy segments for one (source, target) block pair: one whole-block pair
/// when both sides are fully contiguous, otherwise the per-`(layer, outer)`
/// layer-view pairs — the same layout-agnostic decomposition
/// `collect_kv_addresses` uses for the custom kernel. Each entry is
/// `(src_ptr, dst_ptr, size)`; the sizes are uniform across segments.
///
/// SAFETY (the returned raw pointers): the views borrow pool-owned block data
/// that outlives the transfer; the pointers are handed to the CUDA memcpy and
/// never dereferenced on the host.
fn block_copy_segments<Source, Target>(
    src: &Source,
    dst: &mut Target,
) -> Result<Vec<(*const u8, *mut u8, usize)>, TransferError>
where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    let s_data = src.block_data();
    let d_data = dst.block_data_mut();

    if s_data.is_fully_contiguous() && d_data.is_fully_contiguous() {
        let s_view = s_data.block_view()?;
        let mut d_view = d_data.block_view_mut()?;
        debug_assert_eq!(s_view.size(), d_view.size());
        // SAFETY: the views cover pool-owned block data that stays valid
        // until the transfer completes; the pointers are only handed to the
        // driver (never dereferenced by Rust).
        return Ok(vec![(
            unsafe { s_view.as_ptr() },
            unsafe { d_view.as_mut_ptr() },
            s_view.size(),
        )]);
    }

    assert_eq!(
        s_data.num_layers(),
        d_data.num_layers(),
        "segment copy: layer-count mismatch"
    );
    let mut segments = Vec::new();
    for layer_idx in 0..s_data.num_layers() {
        for outer_idx in 0..s_data.num_outer_dims() {
            let s_view = s_data.layer_view(layer_idx, outer_idx)?;
            let mut d_view = d_data.layer_view_mut(layer_idx, outer_idx)?;
            debug_assert_eq!(s_view.size(), d_view.size());
            // SAFETY: the views cover pool-owned block data that stays valid
            // until the transfer completes; the pointers are only handed to
            // the driver (never dereferenced by Rust).
            segments.push((
                unsafe { s_view.as_ptr() },
                unsafe { d_view.as_mut_ptr() },
                s_view.size(),
            ));
        }
    }
    Ok(segments)
}

/// Launch CUDA kernel directly with pinned buffer pointers (no address copying)
unsafe fn launch_copy_kernel_direct(
    src_pinned_ptr: u64,
    dst_pinned_ptr: u64,
    address_count: usize,
    layer_size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {
    // Get kernel function
    let kernel = get_copy_kernel()?;

    tracing::debug!(
        "LAUNCHING KERNEL: {} pairs, src=0x{:x}, dst=0x{:x}",
        address_count,
        src_pinned_ptr,
        dst_pinned_ptr
    );

    let threads_per_block = 256u32;
    let max_blocks = 1024u32;
    let blocks_needed = std::cmp::min(max_blocks, address_count as u32);

    let grid_dim = (blocks_needed, 1, 1);
    let block_dim = (threads_per_block, 1, 1);

    // cuLaunchKernel expects pointers to parameter values
    let src_ptr_param = src_pinned_ptr;
    let dst_ptr_param = dst_pinned_ptr;
    let size_param = layer_size;
    let num_pairs_param = address_count as i32;

    let params = [
        &src_ptr_param as *const _ as *mut std::ffi::c_void,
        &dst_ptr_param as *const _ as *mut std::ffi::c_void,
        &size_param as *const _ as *mut std::ffi::c_void,
        &num_pairs_param as *const _ as *mut std::ffi::c_void,
    ];

    let result = unsafe {
        cudarc::driver::sys::cuLaunchKernel(
            kernel,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            0, // shared memory
            stream.cu_stream(),
            params.as_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut(), // extra
        )
    };

    if result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
        tracing::error!(
            "Kernel launch failed: {:?} - kernel params: {} pairs, layer_size={}, src=0x{:x}, dst=0x{:x}",
            result,
            address_count,
            layer_size,
            src_pinned_ptr,
            dst_pinned_ptr
        );
        return Err(TransferError::ExecutionError(format!(
            "CUDA kernel launch failed: {:?} (address_count={}, layer_size={})",
            result, address_count, layer_size
        )));
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CachedBlockDimensions {
    num_layers: usize,
    num_outer_dims: usize,
    layer_size: usize,
}

/// Key: `(num_layers, num_outer_dims, layer_size)`.
type BlockDimensionsCache = Mutex<HashMap<(usize, usize, usize), CachedBlockDimensions>>;

/// Kernel-dims cache, keyed by the full geometry triple. The historical
/// process-wide single-slot cache froze the FIRST transferred block's
/// dims for the whole process lifetime; with multi-group pools
/// (per-group layouts) a group-1 batch would then launch group-0's
/// frozen `layer_size` and layer count — wrong address/size pairs,
/// corruption. Keying per geometry gives every distinct group geometry
/// its own entry; the map is bounded by the number of distinct
/// geometries, so no eviction is needed. Key collisions across groups
/// are impossible for distinct `(num_layers, num_outer_dims, layer_size)`
/// triples and harmless for identical triples (same dims means the same
/// kernel behavior).
static BLOCK_DIMENSIONS_CACHE: LazyLock<BlockDimensionsCache> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn get_cached_block_dimensions<T: BlockDataProvider>(
    block: &T,
) -> Result<CachedBlockDimensions, TransferError> {
    let dims = calculate_block_dimensions_from_layout(block)?;
    // Poisoning-tolerant lock recovery (the `lora::load_estimator`
    // precedent): the dims are pure layout reads, always safe to cache
    // and use even after another thread's panic.
    let mut cache = BLOCK_DIMENSIONS_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    Ok(*cache
        .entry((dims.num_layers, dims.num_outer_dims, dims.layer_size))
        .or_insert(dims))
}

fn calculate_block_dimensions_from_layout<T: BlockDataProvider>(
    block: &T,
) -> Result<CachedBlockDimensions, TransferError> {
    let block_data = block.block_data();

    // Get dimensions directly from layout (pre-computed values)
    let num_layers = block_data.num_layers();
    let num_outer_dims = block_data.num_outer_dims();
    let layer_size = block_data.layer_view(0, 0).map(|v| v.size()).unwrap_or(0);

    Ok(CachedBlockDimensions {
        num_layers,
        num_outer_dims,
        layer_size,
    })
}

pub fn copy_blocks_with_customized_kernel<'a, Source, Destination>(
    sources: &'a [Source],
    destinations: &'a mut [Destination],
    stream: &CudaStream,
    ctx: &crate::block_manager::block::transfer::TransferContext,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let _context_guard = stream.context().bind_to_thread();
    // Per-batch geometry: the cache is keyed by the CURRENT batch's dims,
    // so every distinct group geometry launches with its own dims (the
    // historical process-wide cache froze the first-ever block's dims —
    // corruption across groups).
    let dims = get_cached_block_dimensions(&sources[0])?;

    // A custom-kernel batch moves ONE group's blocks: every source must
    // share the head block's geometry, or the single-`layer_size` launch
    // would compute wrong address/size pairs (debug-checked; release
    // builds keep the historical trust in batch homogeneity).
    #[cfg(debug_assertions)]
    for (i, src) in sources.iter().enumerate() {
        let d = calculate_block_dimensions_from_layout(src)?;
        debug_assert_eq!(
            (d.num_layers, d.num_outer_dims, d.layer_size),
            (dims.num_layers, dims.num_outer_dims, dims.layer_size),
            "custom-kernel batch mixes geometries (block {i})"
        );
    }

    // Use cached dimensions
    let (src_addresses, dst_addresses) =
        collect_kv_addresses(sources, destinations, dims.num_layers, dims.num_outer_dims)?;

    tracing::debug!(
        "Using vectorized_copy for {} blocks [{}L×{}O×{}B], {} address pairs",
        sources.len(),
        dims.num_layers,
        dims.num_outer_dims,
        dims.layer_size,
        src_addresses.len()
    );

    let size = src_addresses.len() * std::mem::size_of::<u64>();

    let pool = ctx.cuda_mem_pool().ok_or_else(|| {
        TransferError::ExecutionError(
            "TransferContext was not instantiated with a CudaPool; please report this error"
                .to_string(),
        )
    })?;

    // Allocate DEVICE memory from pool (stream-ordered)
    let src_buffer = pool.alloc_async(size, stream).map_err(|e| {
        TransferError::ExecutionError(format!("CUDA pool allocation failed: {}", e))
    })?;
    let dst_buffer = pool.alloc_async(size, stream).map_err(|e| {
        TransferError::ExecutionError(format!("CUDA pool allocation failed: {}", e))
    })?;

    // Copy address buffers from host to device using stream-ordered H2D memcpy
    let result_src = unsafe {
        cuMemcpyHtoDAsync_v2(
            src_buffer,
            src_addresses.as_ptr() as *const std::ffi::c_void,
            size,
            stream.cu_stream(),
        )
    };
    if result_src != CUresult::CUDA_SUCCESS {
        return Err(TransferError::ExecutionError(format!(
            "H2D memcpy for src buffer failed: {:?}",
            result_src
        )));
    }

    let result_dst = unsafe {
        cuMemcpyHtoDAsync_v2(
            dst_buffer,
            dst_addresses.as_ptr() as *const std::ffi::c_void,
            size,
            stream.cu_stream(),
        )
    };
    if result_dst != CUresult::CUDA_SUCCESS {
        return Err(TransferError::ExecutionError(format!(
            "H2D memcpy for dst buffer failed: {:?}",
            result_dst
        )));
    }

    // Record event and synchronize to ensure H2D completes before host vectors drop
    // This is critical: the async H2D memcpy is still reading from src_addresses/dst_addresses
    // host memory when it returns. We must wait for completion before those vectors are dropped.
    let h2d_event = stream
        .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| TransferError::ExecutionError(format!("Failed to record H2D event: {}", e)))?;

    // Launch kernel (reads from device buffers)
    unsafe {
        launch_copy_kernel_direct(
            src_buffer,
            dst_buffer,
            src_addresses.len(),
            dims.layer_size,
            stream,
        )?;
    }

    // Free buffers immediately (stream-ordered - CUDA ensures kernel completes first)
    pool.free_async(src_buffer, stream)
        .map_err(|e| TransferError::ExecutionError(format!("Failed to free src buffer: {}", e)))?;
    pool.free_async(dst_buffer, stream)
        .map_err(|e| TransferError::ExecutionError(format!("Failed to free dst buffer: {}", e)))?;

    // By synchronizing here, we enqueue all the work to the stream, then wait.
    // There is cpu overheads associated with each of those calls.
    // We might as well amortize the transfer of the pointers with those launch overheads.
    h2d_event
        .synchronize()
        .map_err(|e| TransferError::ExecutionError(format!("Failed to sync H2D event: {}", e)))?;

    Ok(())
}

/// Copy a block from a source to a destination using CUDA memcpy
pub fn copy_block<'a, Source, Destination>(
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: &CudaStream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = cuda_memcpy_fn_ptr(&strategy)?;

    #[cfg(debug_assertions)]
    {
        assert_strategy_valid::<Source, Destination>(strategy);
    }

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_view = src_data.block_view()?;
        let mut dst_view = dst_data.block_view_mut()?;

        debug_assert_eq!(src_view.size(), dst_view.size());

        unsafe {
            memcpy_fn(
                src_view.as_ptr(),
                dst_view.as_mut_ptr(),
                src_view.size(),
                stream,
            )?;
        }
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        copy_layers(
            0..src_data.num_layers(),
            sources,
            destinations,
            stream,
            strategy,
        )?;
    }
    Ok(())
}

/// Synchronizes the stream on drop so the pinned bounce never frees
/// under an in-flight DMA (the happy path's explicit event sync covers
/// the normal return; this guard covers every `?`/early-return path).
struct BounceSyncGuard<'a>(&'a CudaStream);

impl Drop for BounceSyncGuard<'_> {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
    }
}

/// Prefix-sum segment offsets plus the image total for a staged
/// LayerSeparate block: `offsets[i]` is segment `i`'s byte offset within
/// the block's bounce image, walked over the ACTUAL per-`(layer, outer)`
/// sizes in layer-major-then-outer order — the same order
/// `block_copy_segments` and `collect_kv_addresses` walk, and the order
/// a `FullyContiguous` image interleaves regions in.
///
/// Equivalence: for uniform segment sizes `s` this reproduces the
/// historical uniform math byte-for-byte — `offsets[i] == i * s` (the
/// old `(layer * num_outer + outer) * (size / seg_count)`) and
/// `total == segment_sizes.len() * s` (the old
/// `layer_view(0, 0) * layers * outer`) — so uniform-geometry groups
/// stage identically to the previous executor.
fn staged_segment_offsets(segment_sizes: &[usize]) -> Result<(Vec<usize>, usize), TransferError> {
    let mut offsets = Vec::with_capacity(segment_sizes.len());
    let mut total = 0usize;
    for &size in segment_sizes {
        offsets.push(total);
        total = total.checked_add(size).ok_or_else(|| {
            TransferError::ExecutionError(format!(
                "staged executor: segment offset overflow: sum of {} segment sizes exceeds usize",
                segment_sizes.len()
            ))
        })?;
    }
    Ok((offsets, total))
}

/// The actual per-`(layer, outer)` layer-view sizes of a block, in
/// layer-major-then-outer order — the same walk `block_copy_segments`
/// and `collect_kv_addresses` use, so the staged DMA legs and the other
/// executors agree on segment order.
fn layer_view_sizes<T: BlockDataProvider>(block: &T) -> Result<Vec<usize>, TransferError> {
    let data = block.block_data();
    let mut sizes = Vec::with_capacity(data.num_layers() * data.num_outer_dims());
    for layer_idx in 0..data.num_layers() {
        for outer_idx in 0..data.num_outer_dims() {
            sizes.push(data.layer_view(layer_idx, outer_idx)?.size());
        }
    }
    Ok(sizes)
}

/// One block's staged-bounce image plan: the block's total image size
/// plus, for LayerSeparate blocks, the actual per-`(layer, outer)`
/// segment sizes and their prefix-sum offsets within the image (empty
/// vectors for fully-contiguous blocks, which stage whole).
struct StagedBlockPlan {
    size: usize,
    segment_sizes: Vec<usize>,
    segment_offsets: Vec<usize>,
}

/// The staged-bounce image plan for one block, computed from the block's
/// ACTUAL layer-view sizes: a fully-contiguous block stages as one
/// whole-block image (`block_view` size); a LayerSeparate block's image
/// is the SUM of its actual per-`(layer, outer)` sizes — not the
/// historical uniform `layer_view(0, 0) * layers * outer` product — so
/// mixed-size layouts keep packing correctly (uniform-geometry groups
/// stay byte-identical; see `staged_segment_offsets`).
fn staged_block_plan<T: BlockDataProvider>(block: &T) -> Result<StagedBlockPlan, TransferError> {
    if block.block_data().is_fully_contiguous() {
        let data = block.block_data();
        return Ok(StagedBlockPlan {
            size: data.block_view()?.size(),
            segment_sizes: Vec::new(),
            segment_offsets: Vec::new(),
        });
    }
    let segment_sizes = layer_view_sizes(block)?;
    let (segment_offsets, size) = staged_segment_offsets(&segment_sizes)?;
    Ok(StagedBlockPlan {
        size,
        segment_sizes,
        segment_offsets,
    })
}

/// Staged batch copy for raw-MMIO host tiers (the BAR-aperture expander
/// path): the driver permits no GPU-side access to the window, so every
/// block stages through a pinned bounce with CPU memcpy legs. Fully
/// synchronous — the whole batch completes before returning (see the
/// `Memcpy` arm's precedent in `handle_local_transfer`).
///
/// Xid-31 safety invariant: no CUDA call ever touches the window VA. The
/// CPU legs move data window↔bounce; the DMAs move data bounce↔device only.
///
/// The DMA legs decompose LayerSeparate device blocks into per-`(layer,
/// outer)` segments (the bounce's logical layout mirrors the window's).
///
/// Bounce allocation tradeoff: `PinnedStorage::new_for_device` allocates
/// WRITE_COMBINED pinned memory by default (unless the
/// `DYN_KVBM_DISABLE_WRITE_COMBINED` env toggle is set). WC is ideal for
/// this staged H2D path (fast CPU writes into the bounce) but penalizes
/// the `CudaStagedD2H` CPU read legs (uncached reads); the non-WC
/// constructor is a perf follow-up if profiling shows the penalty.
pub fn staged_write_blocks_to<Source, Target>(
    sources: &[Source],
    targets: &mut [Target],
    ctx: &Arc<TransferContext>,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    if sources.is_empty() && targets.is_empty() {
        return Ok(());
    }

    if sources.len() != targets.len() {
        return Err(TransferError::CountMismatch(sources.len(), targets.len()));
    }

    #[cfg(debug_assertions)]
    {
        assert_strategy_valid::<Source, Target>(strategy);
    }

    // Per-block staged-bounce image plans, collected once: a
    // fully-contiguous block stages whole (its `block_view` size); a
    // LayerSeparate block's image is the SUM of its actual per-`(layer,
    // outer)` layer-view sizes — byte-identical to the historical uniform
    // `seg * layers * outer` product for uniform-geometry groups, correct
    // for mixed sizes (see `staged_block_plan`).
    let mut plans = Vec::with_capacity(sources.len());
    let mut total = 0usize;
    for src in sources {
        let plan = staged_block_plan(src)?;
        total += plan.size;
        plans.push(plan);
    }

    let stream = ctx.stream();

    // One pinned bounce covering the whole batch, scoped to the executor's
    // device like every other `new_for_device` call site. Per-batch
    // allocation is acceptable for v1; pooling is a follow-up.
    let dev_ord = stream.context().ordinal() as u32;
    let mut bounce = dynamo_memory::PinnedStorage::new_for_device(total, Some(dev_ord))
        .map_err(|e| TransferError::ExecutionError(format!("staged bounce alloc: {e}")))?;
    let _bounce_guard = BounceSyncGuard(stream);
    // SAFETY: `bounce_base` aliases the pinned buffer for the writes below;
    // it stays valid until `bounce` is dropped: the guard's blocking stream
    // synchronize covers every `?`/early-return path and the explicit event
    // syncs below cover the normal return, so the bounce never frees under
    // an in-flight DMA.
    let bounce_base = unsafe { bounce.as_mut_ptr() };

    match strategy {
        TransferStrategy::CudaStagedH2D => {
            // CPU legs: window -> bounce. CPU reads/writes are the only
            // access the raw MMIO window permits (a device-side copy faults
            // with Xid 31), so these legs never touch CUDA.
            let mut offset = 0usize;
            for (idx, src) in sources.iter().enumerate() {
                let size = plans[idx].size;
                let s_view = src.block_data().block_view()?;
                debug_assert_eq!(s_view.size(), size);
                // SAFETY: `s_view` covers `size` readable bytes and the
                // bounce has `total >= offset + size` writable bytes; the
                // regions do not overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(s_view.as_ptr(), bounce_base.add(offset), size)
                };
                offset += size;
            }

            // DMA legs: bounce -> device, async on the stream. When the
            // device block is LayerSeparate, copy per-`(layer, outer)`
            // segment at its prefix-sum offset in the bounce image: the
            // image mirrors the window's FullyContiguous interleaving
            // (layer-major then outer), so each segment lands at its
            // logical offset — byte-identical to the historical
            // `i * (size / seg_count)` offsets for uniform-geometry
            // groups.
            let mut offset = 0usize;
            for (idx, dst) in targets.iter_mut().enumerate() {
                let size = plans[idx].size;
                if dst.block_data().is_fully_contiguous() {
                    let mut d_view = dst.block_data_mut().block_view_mut()?;
                    debug_assert_eq!(d_view.size(), size);
                    // SAFETY: the bounce holds `size` readable bytes at `offset`
                    // and `d_view` covers `size` writable device bytes; the
                    // regions do not overlap.
                    unsafe {
                        cuda_memcpy_h2d(bounce_base.add(offset), d_view.as_mut_ptr(), size, stream)
                    }?;
                } else {
                    // The plan walks the TARGET's actual layer-view sizes
                    // (the window-side plan sized the image; the debug
                    // assert pins the two sides to the same geometry).
                    let plan = staged_block_plan(&*dst)?;
                    debug_assert_eq!(
                        plan.size, size,
                        "staged executor: device block image size != window block size"
                    );
                    let d_data = dst.block_data_mut();
                    let mut seg = 0usize;
                    for layer_idx in 0..d_data.num_layers() {
                        for outer_idx in 0..d_data.num_outer_dims() {
                            let d_view = d_data.layer_view(layer_idx, outer_idx)?;
                            let seg_size = plan.segment_sizes[seg];
                            let seg_off = offset + plan.segment_offsets[seg];
                            seg += 1;
                            // SAFETY: the bounce holds `seg_size` readable
                            // bytes at `seg_off` (the per-block plans sized
                            // it); the pointer is only handed to the driver
                            // (the DMA writes the device bytes, Rust never
                            // dereferences it); the regions do not overlap
                            // (the bounce is host memory, the block is
                            // device memory).
                            unsafe {
                                cuda_memcpy_h2d(
                                    bounce_base.add(seg_off),
                                    d_view.as_ptr() as *mut u8,
                                    seg_size,
                                    stream,
                                )
                            }?;
                        }
                    }
                    debug_assert_eq!(seg, plan.segment_sizes.len());
                }
                offset += size;
            }

            // The async DMAs are still reading the bounce when they return;
            // wait before the bounce drops.
            let event = stream
                .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(|e| {
                    TransferError::ExecutionError(format!("Failed to record staged H2D event: {e}"))
                })?;
            event.synchronize().map_err(|e| {
                TransferError::ExecutionError(format!("Failed to sync staged H2D event: {e}"))
            })?;
        }
        TransferStrategy::CudaStagedD2H => {
            // DMA legs first: device -> bounce, async on the stream. When the
            // device block is LayerSeparate, copy per-`(layer, outer)`
            // segment at its prefix-sum offset in the bounce image: the
            // image mirrors the window's FullyContiguous interleaving
            // (layer-major then outer) — byte-identical to the historical
            // `i * (size / seg_count)` offsets for uniform-geometry
            // groups.
            let mut offset = 0usize;
            for (idx, src) in sources.iter().enumerate() {
                let size = plans[idx].size;
                let s_data = src.block_data();
                if s_data.is_fully_contiguous() {
                    let s_view = s_data.block_view()?;
                    debug_assert_eq!(s_view.size(), size);
                    // SAFETY: `s_view` covers `size` readable device bytes and
                    // the bounce has `total >= offset + size` writable bytes; the
                    // regions do not overlap.
                    unsafe {
                        cuda_memcpy_d2h(s_view.as_ptr(), bounce_base.add(offset), size, stream)
                    }?;
                } else {
                    // The plan was precomputed from THIS source block's
                    // actual layer-view sizes, so the DMA legs consume it
                    // directly.
                    let plan = &plans[idx];
                    let mut seg = 0usize;
                    for layer_idx in 0..s_data.num_layers() {
                        for outer_idx in 0..s_data.num_outer_dims() {
                            let s_view = s_data.layer_view(layer_idx, outer_idx)?;
                            let seg_size = plan.segment_sizes[seg];
                            let seg_off = offset + plan.segment_offsets[seg];
                            seg += 1;
                            // SAFETY: `s_view` covers `seg_size` readable
                            // device bytes and the bounce has
                            // `total >= seg_off + seg_size` writable bytes;
                            // the regions do not overlap.
                            unsafe {
                                cuda_memcpy_d2h(
                                    s_view.as_ptr(),
                                    bounce_base.add(seg_off),
                                    seg_size,
                                    stream,
                                )
                            }?;
                        }
                    }
                    debug_assert_eq!(seg, plan.segment_sizes.len());
                }
                offset += size;
            }

            // The CPU legs must not read the bounce until the DMAs complete.
            let event = stream
                .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(|e| {
                    TransferError::ExecutionError(format!("Failed to record staged D2H event: {e}"))
                })?;
            event.synchronize().map_err(|e| {
                TransferError::ExecutionError(format!("Failed to sync staged D2H event: {e}"))
            })?;

            // CPU legs: bounce -> window. The target window is host memory —
            // CPU writes are exactly where tier-2 data belongs.
            let mut offset = 0usize;
            for (idx, dst) in targets.iter_mut().enumerate() {
                let size = plans[idx].size;
                let mut d_view = dst.block_data_mut().block_view_mut()?;
                debug_assert_eq!(d_view.size(), size);
                // SAFETY: the bounce holds `size` readable bytes at `offset`
                // and `d_view` covers the same writable bytes; the regions do
                // not overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bounce_base.add(offset),
                        d_view.as_mut_ptr(),
                        size,
                    )
                };
                offset += size;
            }
        }
        _ => {
            return Err(TransferError::ExecutionError(
                "staged executor: not a staged strategy".into(),
            ));
        }
    }

    Ok(())
}

/// Direct batch copy for IOMEMORY-registered BAR windows (the
/// `DYN_KVBM_DEVBAR_TRANSFERS=direct` path): the window's device VA is
/// registered (bar-direct-spike-verified, ~3.0 GiB/s both directions on
/// cm008), so every block copies as GPU-side memcpys through the device VA,
/// pointer-translated via the context's io mapping — per-segment
/// (LayerSeparate-capable): one whole-block memcpy when both sides are fully
/// contiguous, otherwise per-`(layer, outer)` layer-view memcpys (the same
/// decomposition `collect_kv_addresses` uses for the custom kernel). Runs on
/// the calling thread; the batch's async memcpys are synced by the closing
/// event (the staged arm's synchronous shape).
///
/// Xid-31 safety: the device VA comes from the registration's device-mapping
/// (the driver-sanctioned surface) — never the raw sysfs VA, which is the
/// copy-kernel fault path.
///
/// The local (KvManagerModel) path does not wire the io mapping —
/// `TRANSFERS=direct` there fails with the guard's clean error; wiring it is
/// a follow-up.
pub fn direct_io_write_blocks_to<Source, Target>(
    sources: &[Source],
    targets: &mut [Target],
    ctx: &Arc<TransferContext>,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Target: BlockDataProviderMut,
{
    let windows = ctx.host_tier_io_mappings();
    if windows.is_empty() {
        return Err(TransferError::ExecutionError(
            "direct-io executor: no io mapping wired (the IOMEMORY registration did not run, or \
             the local KvManagerModel path which does not wire it — direct transfers are \
             supported on the distributed worker path)"
                .into(),
        ));
    }

    if sources.is_empty() && targets.is_empty() {
        return Ok(());
    }
    if sources.len() != targets.len() {
        return Err(TransferError::CountMismatch(sources.len(), targets.len()));
    }

    #[cfg(debug_assertions)]
    {
        assert_strategy_valid::<Source, Target>(strategy);
    }

    let stream = ctx.stream();

    // Sync-on-drop guard for the copy loops below: on a mid-loop memcpy
    // failure the already-issued memcpys are still queued, and returning Err
    // without a sync would let a caller recycling blocks on Err corrupt data
    // under an in-flight DMA. The guard's blocking stream synchronize on drop
    // covers every `?`/early-return path (the staged executor's
    // `BounceSyncGuard` precedent); on the happy path the drop is redundant
    // with the closing event sync below. Reused directly: same-file private
    // guard with exactly these Drop semantics.
    let _io_sync_guard = BounceSyncGuard(stream);

    // Translate a host-tier block pointer into its window's device VA,
    // range-checked against the registered windows. Multi-group tier pools
    // allocate one DISJOINT window per group, so the containing window is
    // unambiguous; a legacy single-group tier has exactly one.
    let translate = |ptr: *const u8, size: usize| -> Result<u64, TransferError> {
        let addr = ptr as usize;
        windows
            .iter()
            .find(|&&(host_base, _, win_len)| {
                addr >= host_base && addr.saturating_add(size) <= host_base.saturating_add(win_len)
            })
            .map(|&(host_base, io_dev_base, _)| io_dev_base + (addr - host_base) as u64)
            .ok_or_else(|| {
                let spans = windows
                    .iter()
                    .map(|&(base, _, len)| format!("[0x{base:x}, 0x{:x})", base + len))
                    .collect::<Vec<_>>()
                    .join(", ");
                TransferError::ExecutionError(format!(
                    "direct-io executor: block [0x{addr:x}, 0x{:x}) outside every registered \
                     window ({spans})",
                    addr.saturating_add(size)
                ))
            })
    };

    match strategy {
        TransferStrategy::CudaDirectIoH2D => {
            // The host tier is the SOURCE: window -> device through the io
            // mapping. Per-segment: the device tier may be LayerSeparate.
            for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                let segments = block_copy_segments(src, dst)?;
                // The per-pair segments must tile the block's whole image
                // (sum to the actual layer-view sizes' total).
                debug_assert_eq!(
                    segments.iter().map(|&(_, _, size)| size).sum::<usize>(),
                    staged_block_plan(src)?.size,
                    "direct-io executor: segments do not sum to the block image size"
                );
                for (s_ptr, d_ptr, seg) in segments {
                    let s_dev = translate(s_ptr, seg)?;
                    // SAFETY: the device block data outlives the transfer
                    // (pool-owned memory); the pointers are only passed to
                    // the driver (never dereferenced by Rust); the regions
                    // do not overlap (the window is host memory, the peer
                    // block is device memory).
                    unsafe {
                        cuda_result::memcpy_dtod_async(d_ptr as u64, s_dev, seg, stream.cu_stream())
                    }
                    .map_err(|e| {
                        TransferError::ExecutionError(format!("direct-io H2D memcpy failed: {e}"))
                    })?;
                }
            }
        }
        TransferStrategy::CudaDirectIoD2H => {
            // The host tier is the TARGET: device -> window through the io
            // mapping. Per-segment: the device tier may be LayerSeparate.
            for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                let segments = block_copy_segments(src, dst)?;
                // The per-pair segments must tile the block's whole image
                // (sum to the actual layer-view sizes' total).
                debug_assert_eq!(
                    segments.iter().map(|&(_, _, size)| size).sum::<usize>(),
                    staged_block_plan(src)?.size,
                    "direct-io executor: segments do not sum to the block image size"
                );
                for (s_ptr, d_ptr, seg) in segments {
                    let d_dev = translate(d_ptr, seg)?;
                    // SAFETY: the device block data outlives the transfer
                    // (pool-owned memory); the pointers are only passed to
                    // the driver (never dereferenced by Rust); the regions
                    // do not overlap (the window is host memory, the peer
                    // block is device memory).
                    unsafe {
                        cuda_result::memcpy_dtod_async(d_dev, s_ptr as u64, seg, stream.cu_stream())
                    }
                    .map_err(|e| {
                        TransferError::ExecutionError(format!("direct-io D2H memcpy failed: {e}"))
                    })?;
                }
            }
        }
        _ => {
            return Err(TransferError::ExecutionError(
                "direct-io executor: not a direct-io strategy".into(),
            ));
        }
    }

    // The async memcpys must complete before the caller proceeds (the staged
    // arm's synchronous shape; the async fire-and-forget is a follow-up).
    let event = stream
        .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| {
            TransferError::ExecutionError(format!("Failed to record direct-io event: {e}"))
        })?;
    event.synchronize().map_err(|e| {
        TransferError::ExecutionError(format!("Failed to sync direct-io event: {e}"))
    })?;

    Ok(())
}

/// Copy a range of layers from a source to a destination using CUDA memcpy
pub fn copy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: &CudaStream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = cuda_memcpy_fn_ptr(&strategy)?;

    #[cfg(debug_assertions)]
    {
        assert_strategy_valid::<Source, Destination>(strategy);
    }

    for layer_idx in layer_range {
        for outer_idx in 0..src_data.num_outer_dims() {
            let src_view = src_data.layer_view(layer_idx, outer_idx)?;
            let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

            debug_assert_eq!(src_view.size(), dst_view.size());

            unsafe {
                memcpy_fn(
                    src_view.as_ptr(),
                    dst_view.as_mut_ptr(),
                    src_view.size(),
                    stream,
                )?;
            }
        }
    }
    Ok(())
}

/// P1 — the span-coalesced bulk copy (the batching plan §4's core win):
/// for a list of block pairs, check if the (layer, outer) segments are
/// MEMORY-CONTIGUOUS across consecutive blocks in BOTH the source and
/// destination pools; if so, issue ONE memcpy per (layer, outer) covering
/// the entire contiguous run instead of per-block. Falls back to the
/// per-block `copy_block` for any non-contiguous gap.
///
/// The key insight (the 2026-10-25 P0 analysis): for LayerSeparate pools
/// (per-layer allocations), consecutive block IDs' layer-L segments ARE
/// contiguous within layer L's pool region — `block[i+1].layer_view(L,O)`
/// == `block[i].layer_view(L,O)` + segment_size. The per-segment fixed
/// cost (~27µs launch + translation) dominates the Coder's ~52K small
/// copies (96% overhead on pinned); coalescing to ~48 per (layer, group)
/// removes it (21× on pinned, 3× on bar-direct).
///
/// SAFETY: the contiguity check compares raw pointers; blocks that pass
/// are verified adjacent in the SAME pool region (the pool's block-ID →
/// address mapping is a simple stride within each per-layer region, and
/// the ID-contiguity implies address-contiguity — the pre-P1 per-block
/// path is byte-identical for any pair that FAILS the check).
pub fn copy_blocks_span_coalesced<'a, Source, Destination>(
    sources: &'a [Source],
    destinations: &'a mut [Destination],
    stream: &CudaStream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    debug_assert_eq!(sources.len(), destinations.len());
    if sources.is_empty() {
        return Ok(());
    }
    let memcpy_fn = cuda_memcpy_fn_ptr(&strategy)?;
    let num_layers = sources[0].block_data().num_layers();
    let num_outer = sources[0].block_data().num_outer_dims();

    #[cfg(debug_assertions)]
    {
        assert_strategy_valid::<Source, Destination>(strategy);
    }

    // Walk the pair list, coalescing contiguous runs and falling back to
    // per-block copies at every gap. A run [i..j) is contiguous when BOTH
    // src and dst have `view[k+1].ptr == view[k].ptr + view[k].size` for
    // all k in [i..j-1) for EVERY (layer, outer) — the per-layer segments
    // within a pool region share a stride, so checking one (L, O) suffices
    // (all layers of the same pool have the same block-ID → address stride).
    let mut i = 0;
    while i < sources.len() {
        // Find the contiguous run starting at i
        let mut j = i + 1;
        while j < sources.len() {
            let src_prev = sources[j - 1].block_data();
            let src_next = sources[j].block_data();
            let dst_prev = destinations[j - 1].block_data();
            let dst_next = destinations[j].block_data();

            // Check the FIRST (layer, outer) for the stride match — all
            // layers of the same pool share the same block stride.
            // SAFETY: the views cover pool-owned data that outlives the
            // pointer arithmetic (the blocks are not written or freed
            // during the check — only address values are compared).
            let (src_contiguous, dst_contiguous) = unsafe {
                let src_prev_ptr = src_prev.layer_view(0, 0)?.as_ptr() as usize;
                let src_next_ptr = src_next.layer_view(0, 0)?.as_ptr() as usize;
                let dst_prev_ptr = dst_prev.layer_view(0, 0)?.as_ptr() as usize;
                let dst_next_ptr = dst_next.layer_view(0, 0)?.as_ptr() as usize;
                (
                    src_next_ptr == src_prev_ptr + src_prev.layer_view(0, 0)?.size(),
                    dst_next_ptr == dst_prev_ptr + dst_prev.layer_view(0, 0)?.size(),
                )
            };

            if !src_contiguous || !dst_contiguous {
                break;
            }
            j += 1;
        }

        let run_len = j - i;
        if run_len == 1 {
            // Single block (or gap start): the per-block path
            cuda::copy_block(&sources[i], &mut destinations[i], stream, strategy)?;
        } else {
            // Contiguous run [i..j): ONE memcpy per (layer, outer) covering
            // all `run_len` blocks — the segments are stride-contiguous.
            let src_data = sources[i].block_data();
            let dst_data = destinations[i].block_data_mut();
            for layer_idx in 0..num_layers {
                for outer_idx in 0..num_outer {
                    let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                    let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;
                    let seg_size = src_view.size();
                    let bulk_size = seg_size * run_len;
                    // SAFETY: the contiguity check proved the span [i..j)
                    // has stride-contiguous memory in both pools for every
                    // (layer, outer) — the bulk pointer is block i's segment
                    // start, and the size covers all run_len blocks. The
                    // per-block path issues the same bytes as run_len
                    // sequential memcpys; the pool memory outlives the
                    // transfer (the stream-ordered completion applies).
                    unsafe {
                        memcpy_fn(src_view.as_ptr(), dst_view.as_mut_ptr(), bulk_size, stream)?;
                    }
                }
            }
            tracing::debug!(
                blocks = run_len,
                layers = num_layers,
                outer = num_outer,
                "span coalesced: {} blocks into {} memcpys (was {})",
                run_len,
                num_layers * num_outer,
                run_len * num_layers * num_outer,
            );
        }
        i = j;
    }
    Ok(())
}

/// The executor-side strategy check for a copy: host-pointer devbar
/// backings (Pinned/Mock) use H2D/D2H; the Vmm device-alias uses D2D
/// (see `TransferContext::host_transfer_adjust`); raw-MMIO Bar/Bdf host
/// tiers use the staged variants (`CudaStagedH2D`/`CudaStagedD2H`), whose
/// CPU legs hand the window VA to no CUDA call, or the direct-io variants
/// (`CudaDirectIoH2D`/`CudaDirectIoD2H`) when the window is
/// IOMEMORY-registered. Asserts membership for devbar pairs, not the exact
/// variant.
#[cfg(debug_assertions)]
fn assert_strategy_valid<Source: StorageTypeProvider, Destination: StorageTypeProvider>(
    strategy: TransferStrategy,
) {
    let devbar_pair = std::any::TypeId::of::<Source::StorageType>()
        == std::any::TypeId::of::<DevbarStorage>()
        || std::any::TypeId::of::<Destination::StorageType>()
            == std::any::TypeId::of::<DevbarStorage>();
    if devbar_pair {
        assert!(
            matches!(
                strategy,
                TransferStrategy::CudaAsyncH2D
                    | TransferStrategy::CudaAsyncD2H
                    | TransferStrategy::CudaAsyncD2D
                    | TransferStrategy::CudaStagedH2D
                    | TransferStrategy::CudaStagedD2H
                    | TransferStrategy::CudaDirectIoH2D
                    | TransferStrategy::CudaDirectIoD2H
            ),
            "devbar pair strategy must be H2D/D2H/D2D/staged/direct-io, got {strategy:?}"
        );
    } else {
        assert_eq!(
            strategy,
            expected_strategy::<Source::StorageType, Destination::StorageType>()
        );
    }
}

/// Helper function to perform the appropriate CUDA memcpy based on storage types
// Allow dead code because it's used in debug assertions
#[allow(dead_code)]
fn expected_strategy<Source: Storage, Dest: Storage>() -> TransferStrategy {
    match (
        std::any::TypeId::of::<Source>(),
        std::any::TypeId::of::<Dest>(),
    ) {
        (src, dst)
            if src == std::any::TypeId::of::<PinnedStorage>()
                && dst == std::any::TypeId::of::<DeviceStorage>() =>
        {
            TransferStrategy::CudaAsyncH2D
        }
        (src, dst)
            if src == std::any::TypeId::of::<DeviceStorage>()
                && dst == std::any::TypeId::of::<PinnedStorage>() =>
        {
            TransferStrategy::CudaAsyncD2H
        }
        (src, dst)
            if src == std::any::TypeId::of::<DevbarStorage>()
                && dst == std::any::TypeId::of::<DeviceStorage>() =>
        {
            // Default (host-pointer backings: Pinned/Mock — Bar/Bdf
            // raw-MMIO tiers stage via the same adjustment): H2D. The Vmm
            // device-alias runs D2D via `TransferContext::host_transfer_adjust`.
            TransferStrategy::CudaAsyncH2D
        }
        (src, dst)
            if src == std::any::TypeId::of::<DeviceStorage>()
                && dst == std::any::TypeId::of::<DevbarStorage>() =>
        {
            // Default: D2H; Vmm runs D2D via the context adjustment.
            TransferStrategy::CudaAsyncD2H
        }
        (src, dst)
            if src == std::any::TypeId::of::<DeviceStorage>()
                && dst == std::any::TypeId::of::<DeviceStorage>() =>
        {
            TransferStrategy::CudaAsyncD2D
        }
        _ => TransferStrategy::Invalid,
    }
}

/// H2D Implementation
#[inline(always)]
unsafe fn cuda_memcpy_h2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source host pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination device pointer is null");

    unsafe {
        let src_slice = std::slice::from_raw_parts(src_ptr, size);
        cuda_result::memcpy_htod_async(dst_ptr as u64, src_slice, stream.cu_stream())
            .map_err(|e| TransferError::ExecutionError(format!("CUDA H2D memcpy failed: {}", e)))?
    };
    Ok(())
}

/// D2H Implementation
#[inline(always)]
unsafe fn cuda_memcpy_d2h(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination host pointer is null");

    unsafe {
        let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, size);
        cuda_result::memcpy_dtoh_async(dst_slice, src_ptr as u64, stream.cu_stream())
            .map_err(|e| TransferError::ExecutionError(format!("CUDA D2H memcpy failed: {}", e)))?;
    }
    Ok(())
}

/// D2D Implementation
#[inline(always)]
unsafe fn cuda_memcpy_d2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination device pointer is null");
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination device memory regions must not overlap for D2D copy"
    );

    unsafe {
        cuda_result::memcpy_dtod_async(dst_ptr as u64, src_ptr as u64, size, stream.cu_stream())
            .map_err(|e| TransferError::ExecutionError(format!("CUDA D2D memcpy failed: {}", e)))?
    };
    Ok(())
}

// Load the vectorized_copy module from FATBIN
fn get_copy_kernel_module() -> Result<cudarc::driver::sys::CUmodule, TransferError> {
    let mut module_guard = COPY_KERNEL_MODULE.lock().unwrap();

    if let Some(module_ptr) = *module_guard {
        return Ok(module_ptr as cudarc::driver::sys::CUmodule);
    }

    // Load the module on first access
    let module = match load_embedded_fatbin() {
        Ok(module) => {
            tracing::debug!("Successfully loaded embedded FATBIN module");
            module
        }
        Err(embedded_err) => {
            tracing::debug!("Embedded FATBIN loading failed: {:?}", embedded_err);
            match load_runtime_fatbin() {
                Ok(module) => {
                    tracing::debug!("Successfully loaded runtime FATBIN module");
                    module
                }
                Err(runtime_err) => {
                    tracing::error!("  Both FATBIN loading methods failed:");
                    tracing::error!("  Embedded error: {:?}", embedded_err);
                    tracing::error!("  Runtime error: {:?}", runtime_err);
                    return Err(TransferError::ExecutionError(
                        "No vectorized_copy FATBIN found (tried embedded and runtime paths)"
                            .to_string(),
                    ));
                }
            }
        }
    };

    let module_ptr = module as usize;
    *module_guard = Some(module_ptr);
    Ok(module as cudarc::driver::sys::CUmodule)
}

// Get the vectorized_copy function
fn get_copy_kernel() -> Result<cudarc::driver::sys::CUfunction, TransferError> {
    let mut func_guard = COPY_KERNEL_FUNCTION.lock().unwrap();

    if let Some(func_ptr) = *func_guard {
        return Ok(func_ptr as cudarc::driver::sys::CUfunction);
    }

    // Load the function on first access
    let module = get_copy_kernel_module()?;
    let func = unsafe {
        let mut func = std::ptr::null_mut();
        let func_name = std::ffi::CString::new("vectorised_copy").unwrap();
        let result =
            cudarc::driver::sys::cuModuleGetFunction(&mut func, module, func_name.as_ptr());
        if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            func
        } else {
            return Err(TransferError::ExecutionError(format!(
                "Failed to get kernel function: {:?}",
                result
            )));
        }
    };

    let func_ptr = func as usize;
    *func_guard = Some(func_ptr);
    Ok(func as cudarc::driver::sys::CUfunction)
}

// Try to load embedded FATBIN (compile-time) - only compiled when FATBIN is available
#[cfg(have_vec_copy_fatbin)]
fn load_embedded_fatbin() -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    // FATBIN was copied to OUT_DIR by build.rs and embedded here
    const FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vectorized_copy.fatbin"));
    tracing::debug!("Loading embedded FATBIN ({} bytes)", FATBIN.len());
    unsafe {
        let mut module = std::ptr::null_mut();
        let result = cudarc::driver::sys::cuModuleLoadData(
            &mut module,
            FATBIN.as_ptr() as *const std::ffi::c_void,
        );
        if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            tracing::debug!("Embedded FATBIN module loaded successfully: {:p}", module);
            return Ok(module);
        } else {
            tracing::error!(
                "Embedded FATBIN cuModuleLoadData failed with CUDA error: {:?}",
                result
            );
        }
    }

    Err(cudarc::driver::DriverError(
        cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND,
    ))
}

// Fallback implementation when FATBIN is not available at compile time
#[cfg(not(have_vec_copy_fatbin))]
fn load_embedded_fatbin() -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    tracing::debug!("No embedded FATBIN available (not compiled with have_vec_copy_fatbin)");
    Err(cudarc::driver::DriverError(
        cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND,
    ))
}

// Try to load FATBIN from filesystem (runtime)
fn load_runtime_fatbin() -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    // 1. Check runtime environment variable first
    if let Ok(runtime_path) = std::env::var(env_cuda::DYN_FATBIN_PATH)
        && let Ok(fatbin_data) = std::fs::read(&runtime_path)
    {
        tracing::debug!("Loading FATBIN from runtime env var: {}", runtime_path);
        unsafe {
            let mut module = std::ptr::null_mut();
            let result = cudarc::driver::sys::cuModuleLoadData(
                &mut module,
                fatbin_data.as_ptr() as *const std::ffi::c_void,
            );
            if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                tracing::debug!("Runtime FATBIN module loaded successfully: {:p}", module);
                return Ok(module);
            } else {
                tracing::error!(
                    "Runtime FATBIN cuModuleLoadData failed with CUDA error: {:?}",
                    result
                );
            }
        }
    }

    // 2. Check standard runtime locations
    let runtime_paths = ["./src/block_manager/block/transfer/kernels/vectorized_copy.fatbin"];

    for path in &runtime_paths {
        if let Ok(fatbin_data) = std::fs::read(path) {
            tracing::debug!("Loading FATBIN from runtime path: {}", path);
            unsafe {
                let mut module = std::ptr::null_mut();
                let result = cudarc::driver::sys::cuModuleLoadData(
                    &mut module,
                    fatbin_data.as_ptr() as *const std::ffi::c_void,
                );
                if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    tracing::debug!(
                        "Runtime path FATBIN module loaded successfully: {:p}",
                        module
                    );
                    return Ok(module);
                } else {
                    tracing::error!(
                        "Runtime path FATBIN cuModuleLoadData failed with CUDA error: {:?}",
                        result
                    );
                }
            }
        } else {
            tracing::debug!("Could not read FATBIN file: {}", path);
        }
    }

    Err(cudarc::driver::DriverError(
        cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND,
    ))
}

#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;
    use crate::block_manager::storage::{
        DeviceAllocator, PinnedAllocator, StorageAllocator, StorageMemset,
    };

    #[test]
    fn test_memset_and_transfer() {
        // Create allocators
        let device_allocator = DeviceAllocator::default();
        let pinned_allocator = PinnedAllocator::default();

        let ctx = device_allocator.ctx().clone();

        // Create CUDA stream
        let stream = ctx.new_stream().unwrap();

        // Allocate host and device memory
        let mut host = pinned_allocator.allocate(1024).unwrap();
        let mut device = device_allocator.allocate(1024).unwrap();

        // Set a pattern in host memory
        StorageMemset::memset(&mut host, 42, 0, 1024).unwrap();

        // Verify host memory was set correctly
        unsafe {
            let ptr = host.as_ptr();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 42));
        }

        // Copy host to device
        unsafe {
            cuda_memcpy_h2d(host.as_ptr(), device.as_mut_ptr(), 1024, stream.as_ref()).unwrap();
        }

        // Synchronize to ensure H2D copy is complete
        stream.synchronize().unwrap();

        // Clear host memory
        StorageMemset::memset(&mut host, 0, 0, 1024).unwrap();

        // Verify host memory was cleared
        unsafe {
            let ptr = host.as_ptr();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 0));
        }

        // Copy back from device to host
        unsafe {
            cuda_memcpy_d2h(device.as_ptr(), host.as_mut_ptr(), 1024, stream.as_ref()).unwrap();
        }

        // Synchronize to ensure D2H copy is complete before verifying
        stream.synchronize().unwrap();

        // Verify the original pattern was restored
        unsafe {
            let ptr = host.as_ptr();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 42));
        }
    }

    // ============================================================================
    // CUDA TRANSFER TESTS FOR LAYOUT COMPATIBILITY
    // ============================================================================

    mod layout_transfer_tests {
        use super::*;
        use crate::block_manager::layout::{
            FullyContiguous, GenericBlockLayout, LayerSeparate, LayoutConfig,
        };

        const TEST_NUM_BLOCKS: usize = 4;
        const TEST_NUM_LAYERS: usize = 3;
        const TEST_OUTER_DIM: usize = 2;
        const TEST_PAGE_SIZE: usize = 8;
        const TEST_INNER_DIM: usize = 16;
        const TEST_DTYPE_WIDTH_BYTES: usize = 2;

        fn create_test_config() -> LayoutConfig {
            LayoutConfig {
                num_blocks: TEST_NUM_BLOCKS,
                num_layers: TEST_NUM_LAYERS,
                outer_dim: TEST_OUTER_DIM,
                page_size: TEST_PAGE_SIZE,
                inner_dim: TEST_INNER_DIM,
                alignment: 256, // GPU-friendly alignment
                dtype_width_bytes: TEST_DTYPE_WIDTH_BYTES,
            }
        }

        /// Test H2D transfers between FullyContiguous host and LayerSeparate device layouts
        #[test]
        fn test_h2d_fc_host_to_ls_device() {
            let device_allocator = DeviceAllocator::default();
            let pinned_allocator = PinnedAllocator::default();
            let ctx = device_allocator.ctx().clone();
            let stream = ctx.new_stream().unwrap();

            let config = create_test_config();

            // Create FullyContiguous host layout
            let host_layout = FullyContiguous::allocate(config.clone(), &pinned_allocator).unwrap();

            // Create LayerSeparate device layout
            let device_layout = LayerSeparate::allocate(config, &device_allocator, true).unwrap();

            // Test data transfer for each memory region
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let device_region = device_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();

                        // Verify regions have same size
                        assert_eq!(
                            host_region.size(),
                            device_region.size(),
                            "Region size mismatch at ({}, {}, {})",
                            block_idx,
                            layer_idx,
                            outer_idx
                        );

                        // Create test pattern
                        let pattern =
                            ((block_idx as u8) << 4) | ((layer_idx as u8) << 2) | (outer_idx as u8);

                        // Fill host memory with pattern
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8,
                                host_region.size(),
                            );
                            host_slice.fill(pattern);
                        }

                        // Transfer H2D
                        unsafe {
                            cuda_memcpy_h2d(
                                host_region.addr() as *const u8,
                                device_region.addr() as *mut u8,
                                host_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                    }
                }
            }

            stream.synchronize().unwrap();

            // Verify transfers by copying back and checking patterns
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let device_region = device_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();

                        let expected_pattern =
                            ((block_idx as u8) << 4) | ((layer_idx as u8) << 2) | (outer_idx as u8);

                        // Create temporary verification buffer
                        let mut verify_buffer =
                            pinned_allocator.allocate(host_region.size()).unwrap();

                        // Copy back from device
                        unsafe {
                            cuda_memcpy_d2h(
                                device_region.addr() as *const u8,
                                verify_buffer.as_mut_ptr(),
                                host_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                        stream.synchronize().unwrap();

                        // Verify pattern
                        unsafe {
                            let verify_slice = std::slice::from_raw_parts(
                                verify_buffer.as_ptr(),
                                host_region.size(),
                            );
                            assert!(
                                verify_slice.iter().all(|&x| x == expected_pattern),
                                "Pattern mismatch at ({}, {}, {}) - expected {}, got {:?}",
                                block_idx,
                                layer_idx,
                                outer_idx,
                                expected_pattern,
                                &verify_slice[0..std::cmp::min(8, verify_slice.len())]
                            );
                        }
                    }
                }
            }
        }

        /// Test D2H transfers from LayerSeparate device to FullyContiguous host
        #[test]
        fn test_d2h_ls_device_to_fc_host() {
            let device_allocator = DeviceAllocator::default();
            let pinned_allocator = PinnedAllocator::default();
            let ctx = device_allocator.ctx().clone();
            let stream = ctx.new_stream().unwrap();

            let config = create_test_config();

            // Create LayerSeparate device layout (block contiguous)
            let device_layout =
                LayerSeparate::allocate(config.clone(), &device_allocator, false).unwrap();

            // Create FullyContiguous host layout
            let host_layout = FullyContiguous::allocate(config, &pinned_allocator).unwrap();

            // Initialize device memory with patterns using a temporary host buffer
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let device_region = device_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let pattern = ((block_idx as u8) << 4)
                            | ((layer_idx as u8) << 2)
                            | (outer_idx as u8)
                            | 0x80;

                        // Create temp buffer with pattern
                        let mut temp_buffer =
                            pinned_allocator.allocate(device_region.size()).unwrap();
                        unsafe {
                            let temp_slice = std::slice::from_raw_parts_mut(
                                temp_buffer.as_mut_ptr(),
                                device_region.size(),
                            );
                            temp_slice.fill(pattern);
                        }

                        // Copy pattern to device
                        unsafe {
                            cuda_memcpy_h2d(
                                temp_buffer.as_ptr(),
                                device_region.addr() as *mut u8,
                                device_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                    }
                }
            }
            stream.synchronize().unwrap();

            // Clear host layout
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8,
                                host_region.size(),
                            );
                            host_slice.fill(0);
                        }
                    }
                }
            }

            // Transfer D2H
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let device_region = device_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let host_region = host_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();

                        unsafe {
                            cuda_memcpy_d2h(
                                device_region.addr() as *const u8,
                                host_region.addr() as *mut u8,
                                device_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                    }
                }
            }
            stream.synchronize().unwrap();

            // Verify patterns in host layout
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let expected_pattern = ((block_idx as u8) << 4)
                            | ((layer_idx as u8) << 2)
                            | (outer_idx as u8)
                            | 0x80;

                        unsafe {
                            let host_slice = std::slice::from_raw_parts(
                                host_region.addr() as *const u8,
                                host_region.size(),
                            );
                            assert!(
                                host_slice.iter().all(|&x| x == expected_pattern),
                                "Pattern mismatch at ({}, {}, {}) - expected {}, got {:?}",
                                block_idx,
                                layer_idx,
                                outer_idx,
                                expected_pattern,
                                &host_slice[0..std::cmp::min(8, host_slice.len())]
                            );
                        }
                    }
                }
            }
        }

        /// Test bidirectional transfers with layout compatibility verification
        #[test]
        fn test_bidirectional_layout_transfers() {
            let device_allocator = DeviceAllocator::default();
            let pinned_allocator = PinnedAllocator::default();
            let ctx = device_allocator.ctx().clone();
            let stream = ctx.new_stream().unwrap();

            let config = create_test_config();

            // Create both layout types
            let host_fc = FullyContiguous::allocate(config.clone(), &pinned_allocator).unwrap();
            let device_ls_outer =
                LayerSeparate::allocate(config.clone(), &device_allocator, true).unwrap();
            let device_ls_block =
                LayerSeparate::allocate(config, &device_allocator, false).unwrap();

            // Test round-trip: Host FC -> Device LS (outer) -> Device LS (block) -> Host FC
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let original_pattern = ((block_idx as u8) << 4)
                            | ((layer_idx as u8) << 2)
                            | (outer_idx as u8)
                            | 0x40;

                        // Step 1: Initialize host FC with pattern
                        let host_region = host_fc
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8,
                                host_region.size(),
                            );
                            host_slice.fill(original_pattern);
                        }

                        // Step 2: Transfer to device LS outer
                        let device_outer_region = device_ls_outer
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        unsafe {
                            cuda_memcpy_h2d(
                                host_region.addr() as *const u8,
                                device_outer_region.addr() as *mut u8,
                                host_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }

                        // Step 3: Transfer between device layouts (D2D)
                        let device_block_region = device_ls_block
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        unsafe {
                            cuda_memcpy_d2d(
                                device_outer_region.addr() as *const u8,
                                device_block_region.addr() as *mut u8,
                                device_outer_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }

                        stream.synchronize().unwrap();

                        // Step 4: Clear host and transfer back
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8,
                                host_region.size(),
                            );
                            host_slice.fill(0);
                        }

                        unsafe {
                            cuda_memcpy_d2h(
                                device_block_region.addr() as *const u8,
                                host_region.addr() as *mut u8,
                                device_block_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                        stream.synchronize().unwrap();

                        // Step 5: Verify pattern survived the round trip
                        unsafe {
                            let host_slice = std::slice::from_raw_parts(
                                host_region.addr() as *const u8,
                                host_region.size(),
                            );
                            assert!(
                                host_slice.iter().all(|&x| x == original_pattern),
                                "Round-trip pattern mismatch at ({}, {}, {}) - expected {}, got {:?}",
                                block_idx,
                                layer_idx,
                                outer_idx,
                                original_pattern,
                                &host_slice[0..std::cmp::min(8, host_slice.len())]
                            );
                        }
                    }
                }
            }
        }

        /// Test transfer performance and alignment impact
        #[test]
        fn test_layout_transfer_alignment_performance() {
            let device_allocator = DeviceAllocator::default();
            let pinned_allocator = PinnedAllocator::default();
            let ctx = device_allocator.ctx().clone();
            let stream = ctx.new_stream().unwrap();

            // Test different alignments
            for alignment in [1, 64, 256, 512] {
                let config = LayoutConfig {
                    num_blocks: 2,
                    num_layers: 2,
                    outer_dim: 1,
                    page_size: 1024,
                    inner_dim: 256,
                    alignment,
                    dtype_width_bytes: 4,
                };

                let host_layout =
                    FullyContiguous::allocate(config.clone(), &pinned_allocator).unwrap();
                let device_layout = FullyContiguous::allocate(config, &device_allocator).unwrap();

                // Measure transfer time (basic timing)
                let start = std::time::Instant::now();

                for block_idx in 0..2 {
                    for layer_idx in 0..2 {
                        let host_region =
                            host_layout.memory_region(block_idx, layer_idx, 0).unwrap();
                        let device_region = device_layout
                            .memory_region(block_idx, layer_idx, 0)
                            .unwrap();

                        unsafe {
                            cuda_memcpy_h2d(
                                host_region.addr() as *const u8,
                                device_region.addr() as *mut u8,
                                host_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                    }
                }
                stream.synchronize().unwrap();

                let duration = start.elapsed();

                // Verify alignment was applied correctly
                let region = host_layout.memory_region(0, 0, 0).unwrap();
                if alignment > 1 {
                    assert_eq!(
                        region.addr() % alignment,
                        0,
                        "Memory not aligned to {} bytes",
                        alignment
                    );
                }

                println!("Transfer with alignment {} took {:?}", alignment, duration);
            }
        }
    }
}

/// CPU-runnable tests (no CUDA device needed, unlike the
/// `testing-cuda`-gated `tests` module above): the kernel-dims cache and
/// the staged segment-plan math exercised against CPU-mocked blocks.
#[cfg(test)]
mod cpu_tests {
    use super::*;
    use crate::block_manager::block::locality::Local;
    use crate::block_manager::block::{BasicMetadata, Block, BlockData};
    use crate::block_manager::layout::{FullyContiguous, LayerSeparate, LayoutConfig};
    use crate::block_manager::storage::tests::{NullDeviceAllocator, NullDeviceStorage};
    use std::sync::Arc;

    fn make_config(num_layers: usize, outer_dim: usize, inner_dim: usize) -> LayoutConfig {
        LayoutConfig {
            num_blocks: 2,
            num_layers,
            outer_dim,
            page_size: 4,
            inner_dim,
            alignment: 1,
            dtype_width_bytes: 2,
        }
    }

    fn fc_block(config: LayoutConfig) -> Block<NullDeviceStorage, Local, BasicMetadata> {
        let layout = FullyContiguous::allocate(config, &NullDeviceAllocator).unwrap();
        let data = BlockData::new(Arc::new(layout), 0, 0, 0);
        Block::new(data, BasicMetadata::default()).unwrap()
    }

    fn ls_block(config: LayoutConfig) -> Block<NullDeviceStorage, Local, BasicMetadata> {
        let layout = LayerSeparate::allocate(config, &NullDeviceAllocator, true).unwrap();
        let data = BlockData::new(Arc::new(layout), 0, 0, 0);
        Block::new(data, BasicMetadata::default()).unwrap()
    }

    /// The kernel-dims cache must key per geometry: a second group's
    /// blocks (different layer count / layer size) must get THEIR dims,
    /// not the first-ever transferred block's frozen dims. This is the
    /// cross-group corruption regression (the historical process-wide
    /// single-slot cache failed here).
    #[test]
    fn block_dimensions_cache_keys_per_geometry() {
        // Group A: 4 layers, 2 outer dims, layer_size = 4 * 64 * 2 = 512.
        let block_a = fc_block(make_config(4, 2, 64));
        // Group B: 8 layers, 1 outer dim, layer_size = 4 * 32 * 2 = 256.
        let block_b = fc_block(make_config(8, 1, 32));

        let dims_a = get_cached_block_dimensions(&block_a).unwrap();
        assert_eq!(
            (dims_a.num_layers, dims_a.num_outer_dims, dims_a.layer_size),
            (4, 2, 512)
        );

        let dims_b = get_cached_block_dimensions(&block_b).unwrap();
        assert_eq!(
            (dims_b.num_layers, dims_b.num_outer_dims, dims_b.layer_size),
            (8, 1, 256),
            "second geometry must not reuse the first block's frozen dims"
        );

        // Repeat lookups stay per-geometry and stable.
        assert_eq!(get_cached_block_dimensions(&block_a).unwrap(), dims_a);
        assert_eq!(get_cached_block_dimensions(&block_b).unwrap(), dims_b);
    }

    /// Uniform segment sizes must plan byte-identically to the
    /// historical uniform math: `offsets[i] == i * seg_size` (the old
    /// `(layer * num_outer + outer) * (size / seg_count)`) and
    /// `total == seg_count * seg_size` (the old
    /// `layer_view(0, 0) * layers * outer`). Uniform-geometry groups
    /// stage exactly as before.
    #[test]
    fn staged_segment_offsets_uniform_equivalence() {
        let seg_size = 512usize;
        for seg_count in [1usize, 2, 6, 13] {
            let sizes = vec![seg_size; seg_count];
            let (offsets, total) = staged_segment_offsets(&sizes).unwrap();
            assert_eq!(total, seg_size * seg_count);
            for (i, &off) in offsets.iter().enumerate() {
                assert_eq!(off, i * seg_size, "uniform offset mismatch at {i}");
            }
        }
    }

    /// Mixed per-segment sizes (the new capability) plan as running
    /// prefix sums; degenerate shapes (no segments, one segment) and the
    /// empty image stay well-defined.
    #[test]
    fn staged_segment_offsets_mixed_sizes() {
        let (offsets, total) = staged_segment_offsets(&[2usize, 3, 5, 7]).unwrap();
        assert_eq!(offsets, vec![0, 2, 5, 10]);
        assert_eq!(total, 17);

        let (offsets, total) = staged_segment_offsets(&[]).unwrap();
        assert!(offsets.is_empty());
        assert_eq!(total, 0);

        let (offsets, total) = staged_segment_offsets(&[9]).unwrap();
        assert_eq!(offsets, vec![0]);
        assert_eq!(total, 9);
    }

    /// The prefix walk must fail (not wrap) when the image size
    /// overflows usize.
    #[test]
    fn staged_segment_offsets_overflow_errors() {
        assert!(staged_segment_offsets(&[usize::MAX, 1]).is_err());
    }

    /// A fully-contiguous block stages whole (no segments, `block_view`
    /// size); a LayerSeparate block of the SAME geometry plans the same
    /// image size (the bounce image mirrors the window's
    /// fully-contiguous layout) and, for a uniform geometry, offsets
    /// byte-identical to the historical uniform math.
    #[test]
    fn staged_block_plan_matches_legacy_uniform_math() {
        // 3 layers x 2 outer dims, segment = 4 * 16 * 2 = 128 bytes.
        let config = || make_config(3, 2, 16);
        let legacy_seg = 128usize;
        let legacy_size = legacy_seg * 3 * 2;

        let fc = fc_block(config());
        let plan = staged_block_plan(&fc).unwrap();
        assert!(plan.segment_sizes.is_empty());
        assert!(plan.segment_offsets.is_empty());
        assert_eq!(plan.size, fc.block_data().block_view().unwrap().size());
        assert_eq!(plan.size, legacy_size);

        let ls = ls_block(config());
        let plan = staged_block_plan(&ls).unwrap();
        assert_eq!(plan.segment_sizes, vec![legacy_seg; 6]);
        assert_eq!(plan.size, legacy_size);
        for (i, &off) in plan.segment_offsets.iter().enumerate() {
            assert_eq!(off, i * legacy_seg, "uniform plan offset mismatch at {i}");
        }
    }

    /// The staged helper's per-segment sizes agree with
    /// `block_copy_segments` (the layout-agnostic decomposition the
    /// direct-io executor and the custom kernel use) — same
    /// layer-major-then-outer walk, same sizes, same total.
    #[test]
    fn layer_view_sizes_agrees_with_block_copy_segments() {
        let config = make_config(3, 2, 16);
        let src = ls_block(config.clone());
        let mut dst = ls_block(config);

        let sizes = layer_view_sizes(&src).unwrap();
        let segments = block_copy_segments(&src, &mut dst).unwrap();
        assert_eq!(sizes.len(), segments.len());
        for (&size, &(_, _, seg_size)) in sizes.iter().zip(segments.iter()) {
            assert_eq!(size, seg_size);
        }
        assert_eq!(
            sizes.iter().sum::<usize>(),
            staged_block_plan(&src).unwrap().size
        );
    }
}

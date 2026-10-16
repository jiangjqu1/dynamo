// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::zmq::*;
use async_trait::async_trait;
use transfer::*;
use utils::*;

use crate::block_manager::{
    BasicMetadata, BlockMetadata, LayoutConfigBuilder, NixlLayout, Storage,
    block::{
        Block, layout_to_blocks, locality,
        transfer::{PoolConfig, TransferContext},
    },
    connector::scheduler::TransferSchedulerClient,
    layout::LayoutType,
    offload::{max_concurrent_transfers, max_transfer_batch_size},
    storage::{
        DevbarAllocator, DeviceAllocator, DeviceStorage, DiskAllocator, DiskStorage,
        StorageAllocator, torch::TorchTensor,
    },
};

use derive_builder::Builder;
use nixl_sys::Agent as NixlAgent;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use validator::Validate;

use tokio::runtime::Handle;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tokio::sync::{Mutex, RwLock, oneshot};

struct WorkerState {
    ready_for_ping: AtomicBool,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            ready_for_ping: AtomicBool::new(false),
        }
    }
    fn mark_ready(&self) {
        self.ready_for_ping.store(true, Ordering::SeqCst);
    }
    fn is_ready(&self) -> bool {
        self.ready_for_ping.load(Ordering::SeqCst)
    }
}

pub fn load_and_validate_tensors(
    tensors: &[Arc<dyn TorchTensor>],
    device_id: usize,
    groups: &[LayerGroup],
) -> anyhow::Result<(Vec<DeviceStorage>, Vec<usize>)> {
    let mut shape = None;

    let mut device_tensors = Vec::with_capacity(tensors.len());
    let allocator = DeviceAllocator::new(device_id)?;

    for tensor in tensors {
        // Check the stride, and ensure our tensor is contiguous.
        // TODO: We eventually need to be able to handle this.
        let stride = tensor.stride();
        tracing::debug!("stride: {:?}", stride);
        tracing::debug!("stride is monotonically decreasing for NHD layout");
        tracing::debug!("stride is NOT monotonically decreasing for HND layout");

        // Legacy contract (no groups): all layer tensors share one shape.
        // With groups, shapes may differ across groups and are checked
        // per-group below.
        if groups.is_empty() {
            // Check that all layer tensors have the same shape.
            // TODO: We eventually need to support the weirder models with heterogenous layers.
            if let Some(shape) = shape.as_ref() {
                if *shape != tensor.shape() {
                    return Err(anyhow::anyhow!(
                        "All tensors must have the same shape! Got {:?} and {:?}",
                        *shape,
                        tensor.shape()
                    ));
                }
            } else {
                shape = Some(tensor.shape());
            }
        }

        // Build the storage object from the tensor.
        let device_tensor = DeviceStorage::new_from_torch(allocator.ctx(), tensor.clone())?;

        device_tensors.push(device_tensor);
    }

    // Grouped contract: tensors within a group share a shape; across groups
    // they may differ (e.g. attention K/V pages vs dense mamba state rows).
    for (group_idx, group) in groups.iter().enumerate() {
        let mut group_shape: Option<Vec<usize>> = None;
        for &tensor_idx in &group.tensor_indices {
            let tensor = tensors.get(tensor_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "Layer group {} references tensor index {} but only {} tensors are registered",
                    group_idx,
                    tensor_idx,
                    tensors.len()
                )
            })?;
            let tensor_shape = tensor.shape();
            match group_shape {
                Some(prev) if prev != tensor_shape => {
                    return Err(anyhow::anyhow!(
                        "All tensors in layer group {} must have the same shape! Got {:?} and {:?}",
                        group_idx,
                        prev,
                        tensor_shape
                    ));
                }
                None => group_shape = Some(tensor_shape),
                _ => {}
            }
        }
    }

    let shape = shape
        .or_else(|| tensors.first().map(|t| t.shape()))
        .ok_or_else(|| anyhow::anyhow!("empty tensor list: no tensors were provided"))?;
    Ok((device_tensors, shape))
}

fn build_agent(worker_id: usize, use_gds: bool) -> anyhow::Result<NixlAgent> {
    let agent = NixlAgent::new(&format!("kvbm-worker-{}", worker_id))?;
    // POSIX and GDS_MT only serve files. Host and device layouts need a backend
    // that covers DRAM and VRAM, so create UCX first, matching the v2 default
    // backend set in v2/physical/transfer/nixl_agent.
    match agent.get_plugin_params("UCX") {
        Ok((_, ucx_params)) => {
            agent.create_backend("UCX", &ucx_params)?;
        }
        Err(_) => {
            tracing::warn!("No UCX plugin found; host and device NIXL registration will fail");
        }
    }
    if use_gds {
        let (_, gds_params) = agent.get_plugin_params("GDS_MT")?;
        agent.create_backend("GDS_MT", &gds_params)?;
    }
    let (_, posix_params) = agent.get_plugin_params("POSIX")?;
    agent.create_backend("POSIX", &posix_params)?;

    Ok(agent)
}

// Helper: perform allocation and build transfer handler (factored from previous code)
async fn perform_allocation_and_build_handler(
    group_layouts: Vec<GroupLayoutSpec>,
    worker_config: KvbmWorkerConfig,
    leader_meta: LeaderMetadata,
    worker_id: usize,
    device_id: usize,
    scheduler_client: Option<TransferSchedulerClient>,
) -> anyhow::Result<BlockTransferHandler> {
    let num_groups = group_layouts.len();
    // Determine if this rank should allocate G2/G3 (host/disk)
    // - Sharded mode (rank=None): all ranks allocate
    // - Replicated mode (rank=Some(r)): only rank 0 allocates
    let should_allocate_offload = match worker_config.rank {
        None => true,     // Sharded mode: all ranks allocate
        Some(0) => true,  // Replicated mode rank 0: allocate
        Some(_) => false, // Replicated mode non-rank0: skip
    };

    if !should_allocate_offload {
        tracing::info!(
            "Rank {} skipping host/disk allocation (replicated mode)",
            worker_config.rank.unwrap_or(-1)
        );
    }

    // Only create NIXL agent if we need disk blocks AND we should allocate
    let need_disk = should_allocate_offload && leader_meta.num_disk_blocks > 0;
    let agent = build_agent(worker_id, need_disk)?;
    let pool_config = PoolConfig {
        enable_pool: true,
        max_concurrent_transfers: max_concurrent_transfers(),
        max_transfer_batch_size: max_transfer_batch_size(),
        // Sizing-only bound for the context's CUDA pool pre-warm: the
        // largest single-request layer/outer count across groups (exactly
        // the single group's values for the legacy path).
        num_outer_components: group_layouts
            .iter()
            .map(|spec| spec.geometry.outer_dim)
            .max()
            .unwrap_or(1),
        num_layers: group_layouts
            .iter()
            .map(|spec| spec.geometry.num_layers)
            .max()
            .unwrap_or(1),
    };
    // Host (G2 / tier-2) allocator, built before the transfer context so its
    // backing's transfer_semantics() can set the context's host-transfer
    // semantics: DeviceAlias for a VMM region, StagedCpu for a Bar/Bdf region
    // on a raw sysfs MMIO aperture (staged default; `direct` opt-in),
    // HostPointer otherwise. The per-group pools are only allocated below
    // when should_allocate_offload — sequentially from this ONE allocator,
    // so Bar/Bdf apertures carve disjoint windows per group.
    let host_allocator = Arc::new(DevbarAllocator::from_env()?);
    let host_transfer_semantics = host_allocator.transfer_semantics();
    let mut transfer_context = TransferContext::new(
        Arc::new(Some(agent)),
        DeviceAllocator::new(device_id)?.ctx().new_stream()?,
        Handle::current(),
        Some(pool_config),
    )
    .map_err(|e| {
        anyhow::anyhow!(
            "Failed to create transfer context for worker {} with CUDA memory pool: {}. \
             This is a critical error - the worker cannot start without CUDA memory pools. \
             Please ensure sufficient GPU memory is available on device {}.",
            worker_id,
            e,
            device_id
        )
    })?;
    transfer_context.set_host_transfer_semantics(host_transfer_semantics);
    let transfer_context = Arc::new(transfer_context);

    // disk (G3) allocator: created once when the tier is enabled, shared by
    // the per-group allocates (each allocate creates its own file-backed
    // storage — no aliasing).
    let disk_allocator = if should_allocate_offload && leader_meta.num_disk_blocks > 0 {
        Some(Arc::new(DiskAllocator::from_env()?))
    } else {
        None
    };

    // One pool list per group, per tier. Block-set ids: the legacy
    // single-group path keeps the historical 0/1/2; grouped tiers namespace
    // per group (device 16+g, host 32+g, disk 48+g) so every group's
    // serialized layout gets its own NixlBlockSet entry per worker.
    let mut device_blocks = Vec::with_capacity(num_groups);
    let mut host_blocks = Vec::with_capacity(num_groups);
    let mut disk_blocks = Vec::with_capacity(num_groups);
    // Each group's host window carries its OWN io mapping (Bar/Bdf direct
    // transfers); collected per allocate and wired as a set below.
    let mut host_io_mappings: Vec<(usize, u64, usize)> = Vec::new();

    for (group_idx, spec) in group_layouts.into_iter().enumerate() {
        let GroupLayoutSpec {
            geometry,
            device_layout,
        } = spec;

        // device - always allocated on all ranks
        let device_set = if num_groups == 1 { 0 } else { 16 + group_idx };
        device_blocks.push(Some(KvbmWorker::make_layout::<_, BasicMetadata>(
            device_layout,
            transfer_context.nixl_agent().as_ref(),
            device_set,
            worker_id,
        )?));

        // host (G2 / tier-2) - only allocated if should_allocate_offload.
        // Backed by a BAR-mapped devbar region when DYN_KVBM_DEVBAR_* is
        // configured, otherwise standard pinned host memory (default
        // deployments unchanged). Grouped contract: FullyContiguous over ONE
        // allocation per group at the shared num_host_blocks count, from the
        // shared DevbarAllocator.
        if should_allocate_offload && leader_meta.num_host_blocks > 0 {
            let host_layout_type = if num_groups == 1 {
                worker_config.host_layout_type
            } else {
                LayoutType::FullyContiguous
            };
            // The layout takes the Arc by value; clone so the io-mapping
            // stash (filled by the allocate inside) stays readable after
            // the call.
            let host_alloc_wire = Arc::clone(&host_allocator);
            let host_layout = tier_layout_builder(&geometry, leader_meta.num_host_blocks)
                .build()?
                .allocate_layout(host_layout_type, host_alloc_wire)
                .map_err(|e| anyhow::anyhow!("host layout for layer group {group_idx}: {e}"))?;
            if let Some(mapping) = host_allocator.io_mapping() {
                host_io_mappings.push(mapping);
            }
            let host_set = if num_groups == 1 { 1 } else { 32 + group_idx };
            host_blocks.push(Some(KvbmWorker::make_layout::<_, BasicMetadata>(
                host_layout,
                transfer_context.nixl_agent().as_ref(),
                host_set,
                worker_id,
            )?));
        } else {
            host_blocks.push(None);
        }

        // disk (G3) - only allocated if should_allocate_offload
        if let (true, Some(disk_allocator)) = (should_allocate_offload, disk_allocator.as_ref()) {
            let disk_layout_type = if num_groups == 1 {
                worker_config.disk_layout_type
            } else {
                LayoutType::FullyContiguous
            };
            let disk_alloc_wire: Arc<dyn StorageAllocator<DiskStorage>> = disk_allocator.clone();
            let disk_layout = tier_layout_builder(&geometry, leader_meta.num_disk_blocks)
                .build()?
                .allocate_layout(disk_layout_type, disk_alloc_wire)
                .map_err(|e| anyhow::anyhow!("disk layout for layer group {group_idx}: {e}"))?;
            let disk_set = if num_groups == 1 { 2 } else { 48 + group_idx };
            disk_blocks.push(Some(KvbmWorker::make_layout::<_, BasicMetadata>(
                disk_layout,
                transfer_context.nixl_agent().as_ref(),
                disk_set,
                worker_id,
            )?));
        } else {
            disk_blocks.push(None);
        }
    }

    // Direct transfers: wire the BAR windows' io mappings (host base,
    // device base, length) into the context for the executor's pointer
    // translation — one entry per group's window. Present only when the
    // IOMEMORY registration succeeded.
    if !host_io_mappings.is_empty() {
        transfer_context.set_host_tier_io_mappings(host_io_mappings);
    }

    let handler = BlockTransferHandler::new(
        device_blocks,
        host_blocks,
        disk_blocks,
        transfer_context,
        scheduler_client,
        worker_config.nccl_config,
    )?;
    Ok(handler)
}

struct WorkerMetadataHandler {
    num_device_blocks: usize,
    bytes_per_block: usize,
    /// One entry per HMA group, in registration order; empty only for a
    /// legacy caller that never derived per-group geometries.
    bytes_per_block_per_group: Vec<usize>,
}

#[async_trait]
impl Handler for WorkerMetadataHandler {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
        let payload = bincode::serde::encode_to_vec(
            &WorkerMetadata {
                num_device_blocks: self.num_device_blocks,
                bytes_per_block: self.bytes_per_block,
                bytes_per_block_per_group: self.bytes_per_block_per_group.clone(),
            },
            bincode::config::standard(),
        )?;
        message
            .reply(ZMQ_WORKER_METADATA_MESSAGE, &[payload])
            .await?;
        Ok(())
    }
}

// Leader sends allocation config -> allocate -> publish handler -> mark ready -> ACK
struct LeaderMetadataHandler {
    state: Arc<WorkerState>,
    /// Per-group device layout specs (one entry per HMA group; exactly one
    /// for the legacy path), consumed once by the allocation task.
    group_layouts: Mutex<Option<Vec<GroupLayoutSpec>>>,
    worker_config: KvbmWorkerConfig,
    worker_id: usize,
    device_id: usize,
    scheduler_client: Option<TransferSchedulerClient>,
    handler_cell: Arc<RwLock<Option<BlockTransferHandler>>>,
    handler_tx: Arc<Mutex<Option<oneshot::Sender<BlockTransferHandler>>>>,
    started: AtomicBool,
}

#[async_trait]
impl Handler for LeaderMetadataHandler {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
        // Always ACK ASAP so Drop can't panic and leader can finish the round.
        if let Err(e) = message.ack().await {
            tracing::error!("leader_metadata: failed to ACK: {e:#}");
        }

        // Validate payload; if bad, ignore.
        if message.data.len() != 1 {
            tracing::error!(
                "leader_metadata expects 1 payload frame (got {})",
                message.data.len()
            );
            return Ok(());
        }
        let leader_meta: LeaderMetadata = match bincode::serde::decode_from_slice(
            &message.data[0],
            bincode::config::standard(),
        ) {
            Ok((m, _)) => m,
            Err(e) => {
                tracing::error!("leader_metadata: bad payload: {e:#}");
                return Ok(());
            }
        };

        // Single-flight: only the first message triggers allocation.
        if self
            .started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            tracing::debug!("leader_metadata: allocation already started; dropping duplicate");
            return Ok(());
        }

        // Take the per-group layout specs once.
        let group_layouts = {
            let mut guard = self.group_layouts.lock().await;
            match guard.take() {
                Some(d) => d,
                None => {
                    tracing::warn!("leader_metadata: group_layouts already consumed; dropping");
                    return Ok(());
                }
            }
        };

        // Capture what we need and run allocation in the background.
        let worker_config = self.worker_config.clone();
        let worker_id = self.worker_id;
        let device_id = self.device_id;
        let scheduler_client = self.scheduler_client.clone();
        let handler_cell = self.handler_cell.clone();
        let handler_tx = self.handler_tx.clone();
        let state = self.state.clone();

        tokio::spawn(async move {
            match perform_allocation_and_build_handler(
                group_layouts,
                worker_config,
                leader_meta,
                worker_id,
                device_id,
                scheduler_client,
            )
            .await
            {
                Ok(handler) => {
                    // Install transfer handler
                    {
                        let mut w = handler_cell.write().await;
                        *w = Some(handler.clone());
                    }
                    // Return handler to creator (once)
                    {
                        let mut g = handler_tx.lock().await;
                        if let Some(tx) = g.take() {
                            let _ = tx.send(handler);
                        }
                    }
                    // Now the worker can ACK pings
                    state.mark_ready();
                    tracing::info!("allocation finished; worker is ping-ACK-able");
                }
                Err(e) => {
                    tracing::error!("allocation failed: {e:#}");
                    // leave ready=false so pings keep being ignored
                }
            }
        });

        Ok(())
    }
}

// Gated ping: the worker can only response to ping after the state is ready
struct GatedPing {
    state: Arc<WorkerState>,
    // fired exactly once after the first successful ping ACK
    layout_ready_tx: Mutex<Option<oneshot::Sender<String>>>,
}

#[async_trait]
impl Handler for GatedPing {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
        if !self.state.is_ready() {
            tracing::info!(
                "KVBM worker is under initialization. It could take a while if set with large CPU or DISK cache size. Please wait..."
            );
            tracing::debug!("Ping received but worker not ready; deferring ACK");
            // Prevent Drop panic; leader won't get an ACK for this round and will retry.
            message.mark_handled();
            return Ok(());
        }

        message.ack().await?;

        // After a successful ACK, flip the readiness oneshot exactly once
        let mut guard = self.layout_ready_tx.lock().await;
        if let Some(tx) = guard.take() {
            let _ = tx.send("ping-acked".to_string());
            tracing::info!("Reported ping-ready after first ACK");
        }

        Ok(())
    }
}

// Transfer dispatcher that waits until block transfer handler exists
struct BlockTransferDispatch {
    cell: Arc<RwLock<Option<BlockTransferHandler>>>,
}

#[async_trait]
impl Handler for BlockTransferDispatch {
    async fn handle(&self, message: MessageHandle) -> anyhow::Result<()> {
        let maybe = { self.cell.read().await.clone() };
        if let Some(inner) = maybe {
            inner.handle(message).await
        } else {
            Err(anyhow::anyhow!("transfer handler not ready yet"))
        }
    }
}

fn validate_page_size(value: usize) -> Result<(), validator::ValidationError> {
    // Power-of-two pages are the historical contract. Pages whose size is a
    // multiple of 16 (e.g. 784 = 16 x 49 tokens for hybrid attention/mamba
    // models) tile the same way and are accepted as well. Zero is never a
    // valid page size.
    if value == 0 || (!value.is_power_of_two() && !value.is_multiple_of(16)) {
        return Err(validator::ValidationError::new(
            "page_size_not_power_of_two",
        ));
    }
    Ok(())
}

/// Geometry of one group of KV layers within a worker's registered tensors.
///
/// A group is a set of layers sharing one physical page geometry, e.g. the
/// standard attention K/V tensors versus a mamba block's dense state rows.
/// `page_bytes` is the value the layout multiplies as its `page_size`
/// (page-token count for attention groups; for mamba groups the dense int8
/// row size `C` with `outer = inner = dtype = 1`, so the layout's
/// `page * inner * dtype` product reproduces `C` without re-multiplication).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGroup {
    pub num_layers: usize,
    pub page_bytes: usize,
    pub outer_dim: usize,
    pub inner_dim: usize,
    pub dtype_width_bytes: usize,
    pub is_mamba: bool,
    /// Indices into the registration-order tensor list; strictly
    /// increasing within the group, and across all groups an exact
    /// partition of `0..tensors.len()` (each index claimed exactly once;
    /// cross-group order is free, so interleaved groups are valid).
    pub tensor_indices: Vec<usize>,
}

impl LayerGroup {
    /// Attention group: `page_tokens` per block row, `outer_dim` K/V
    /// components of `inner_dim` elements each, `dtype_width_bytes` bytes
    /// per element.
    pub fn attention(
        num_layers: usize,
        page_tokens: usize,
        outer_dim: usize,
        inner_dim: usize,
        dtype_width_bytes: usize,
    ) -> Self {
        Self {
            num_layers,
            page_bytes: page_tokens,
            outer_dim,
            inner_dim,
            dtype_width_bytes,
            is_mamba: false,
            tensor_indices: Vec::new(),
        }
    }

    /// Mamba group: `num_layers` dense int8 `[B, C]` state rows of `C` bytes
    /// per layer. Geometry is fully determined: `outer = inner = dtype = 1`
    /// and `page_bytes = C`.
    pub fn mamba(num_layers: usize, page_bytes: usize) -> Self {
        Self {
            num_layers,
            page_bytes,
            outer_dim: 1,
            inner_dim: 1,
            dtype_width_bytes: 1,
            is_mamba: true,
            tensor_indices: Vec::new(),
        }
    }

    /// Synthesize the single legacy group for a config without explicit
    /// `groups`: `page_size` is the page-token count and `num_tensors` the
    /// number of registered tensors (one index per tensor in registration
    /// order; for FullyContiguous layouts a single tensor may hold every
    /// layer).
    pub fn from_legacy(
        num_layers: usize,
        page_size: usize,
        outer_dim: usize,
        inner_dim: usize,
        dtype_width_bytes: usize,
        num_tensors: usize,
    ) -> Self {
        Self {
            num_layers,
            page_bytes: page_size,
            outer_dim,
            inner_dim,
            dtype_width_bytes,
            is_mamba: false,
            tensor_indices: (0..num_tensors).collect(),
        }
    }

    /// Total physical bytes of one block across the group's layers:
    /// `num_layers * outer_dim * page_bytes * inner_dim * dtype_width_bytes`.
    pub fn bytes_per_block(&self) -> usize {
        self.num_layers * self.outer_dim * self.page_bytes * self.inner_dim * self.dtype_width_bytes
    }

    pub fn is_mamba(&self) -> bool {
        self.is_mamba
    }

    pub fn layout_type(&self) -> LayoutType {
        LayoutType::LayerSeparate {
            outer_contiguous: true,
        }
    }
}

/// Derived per-group geometry for the layout/pool path (staging for
/// per-group layouts; pool construction currently consumes only the first
/// group).
#[derive(Debug, Clone, Copy)]
struct GroupGeometry {
    layout_type: LayoutType,
    num_layers: usize,
    outer_dim: usize,
    inner_dim: usize,
    page_size: usize,
    dtype_width_bytes: usize,
    bytes_per_block: usize,
}

/// Infer `(outer_dim, inner_dim)` from a LayerSeparate tensor shape.
///
/// Shared by the legacy single-group path and per-group derivation; the
/// inference rules are unchanged from the historical inline version.
fn infer_layer_separate_dims(
    shape: &[usize],
    page_size: usize,
    outer_contiguous: bool,
    explicit: (Option<usize>, Option<usize>),
) -> (usize, usize) {
    // Explicit dims provided by caller (e.g. Python via KvTensorLayout) — use as-is.
    if let (Some(od), Some(id)) = explicit {
        return (od, id);
    }
    // No explicit dims: infer from shape.
    // outer_dim valid range is [1, 2] (1 = MLA fused, 2 = standard K/V split).
    // If the candidate dimension exceeds 2 the tensor has no explicit K/V axis
    // (e.g. MLA models produce [n_blocks, page_size, latent_dim]) — fall back to
    // outer_dim=1 and compute inner_dim from all dims after n_blocks.
    let candidate = if outer_contiguous { shape[0] } else { shape[1] };
    if candidate <= 2 {
        // Standard layout: outer_dim is encoded in the shape.
        //   outer_contiguous=true:  [outer_dim, n_blocks, page_size, inner_dim]
        //   outer_contiguous=false: [n_blocks, outer_dim, page_size, inner_dim]
        let inner_dim = shape[2..].iter().product::<usize>() / page_size;
        (candidate, inner_dim)
    } else {
        // MLA-style: no explicit K/V split, treat as outer_dim=1.
        let dims_start = if outer_contiguous { 2 } else { 1 };
        let inner_dim = shape[dims_start..].iter().product::<usize>() / page_size;
        (1, inner_dim)
    }
}

/// One HMA group's layout spec: the group's derived geometry plus its DEVICE
/// layout, built over the group's own tensor subset. Per-group tier (host /
/// disk) layouts are allocated from the same geometry at pool-construction
/// time; one spec exists per group, exactly one for the legacy path.
struct GroupLayoutSpec {
    geometry: GroupGeometry,
    device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
}

/// Handshake-reported geometry: the scalar Σ and the per-group vec the
/// worker publishes in `WorkerMetadata`.
struct HandshakeGeometry {
    bytes_per_block: usize,
    bytes_per_block_per_group: Vec<usize>,
}

/// A `LayoutConfig` builder for a group's geometry at `num_blocks` — the
/// per-group tier (host / disk) layout configuration.
fn tier_layout_builder(geometry: &GroupGeometry, num_blocks: usize) -> LayoutConfigBuilder {
    let mut builder = LayoutConfigBuilder::default();
    builder
        .num_layers(geometry.num_layers)
        .outer_dim(geometry.outer_dim)
        .page_size(geometry.page_size)
        .inner_dim(geometry.inner_dim)
        .dtype_width_bytes(geometry.dtype_width_bytes)
        .num_blocks(num_blocks);
    builder
}

/// Derive one group's geometry. Mamba groups carry their full geometry (no
/// shape inference); attention groups reuse the LayerSeparate inference
/// scoped to the group's first registered tensor.
///
/// Precondition: `config.validate()` ensured every group has non-empty,
/// in-range `tensor_indices`.
fn derive_group_geometry(
    group: &LayerGroup,
    tensors: &[Arc<dyn TorchTensor>],
    explicit: (Option<usize>, Option<usize>),
    num_device_blocks: usize,
) -> GroupGeometry {
    debug_assert!(
        !group.tensor_indices.is_empty(),
        "derive_group_geometry precondition: group has tensor indices (validate() enforces this)"
    );
    debug_assert!(
        group.tensor_indices.iter().all(|&i| i < tensors.len()),
        "derive_group_geometry precondition: tensor indices in range (validate() enforces this)"
    );
    if group.is_mamba() {
        GroupGeometry {
            layout_type: group.layout_type(),
            num_layers: group.num_layers,
            outer_dim: group.outer_dim,
            inner_dim: group.inner_dim,
            page_size: group.page_bytes,
            dtype_width_bytes: group.dtype_width_bytes,
            bytes_per_block: group.bytes_per_block(),
        }
    } else {
        let first_idx = group.tensor_indices[0];
        let shape = tensors[first_idx].shape();
        // The block-axis convention from the ACTUAL shape (the same rule as
        // layer_separate_auto): a leading num_blocks axis ([num_blocks, ...])
        // means outer_contiguous=false; a small leading axis ([outer_dim,
        // num_blocks, ...], e.g. K/V-outer [2, num_blocks, ...] or MLA
        // [1, num_blocks, ...]) means true. vLLM 0.29's LBNHC cache is
        // [num_blocks, kv_heads, block_size, 2*head_size] — the FALSE
        // convention. Hard-coding true derived inner_dim from shape[2..]
        // instead of shape[1..] for LBNHC shapes (inner=128 vs 1024 for the
        // 0.6B), truncating every stored block to 1/8th of its bytes (the
        // determinism root cause: restored KV 87.5% missing).
        let outer_contiguous = shape.first().copied().unwrap_or(0) < num_device_blocks;
        let (outer_dim, inner_dim) =
            infer_layer_separate_dims(&shape, group.page_bytes, outer_contiguous, explicit);
        GroupGeometry {
            layout_type: LayoutType::LayerSeparate { outer_contiguous },
            num_layers: group.num_layers,
            outer_dim,
            inner_dim,
            page_size: group.page_bytes,
            dtype_width_bytes: group.dtype_width_bytes,
            bytes_per_block: group.num_layers
                * outer_dim
                * group.page_bytes
                * inner_dim
                * group.dtype_width_bytes,
        }
    }
}

#[derive(Builder, Clone, Validate)]
#[builder(pattern = "owned")]
pub struct KvbmWorkerConfig {
    cancel_token: CancellationToken,

    num_device_blocks: usize,

    #[validate(custom(function = "validate_page_size"), range(max = 1024))]
    #[builder(default = "32")]
    page_size: usize,

    #[builder(default = "Vec::new()")]
    tensors: Vec<Arc<dyn TorchTensor>>,

    #[builder(default = "0")]
    device_id: usize,

    #[builder(default = "2")]
    dtype_width_bytes: usize,

    #[builder(default = "LayoutType::FullyContiguous")]
    device_layout_type: LayoutType,

    #[builder(default = "LayoutType::FullyContiguous")]
    host_layout_type: LayoutType,

    #[builder(default = "LayoutType::FullyContiguous")]
    disk_layout_type: LayoutType,

    /// Explicit outer dimension (1 for MLA, 2 for standard K/V).
    /// When set, skips shape inference. Python should always provide this.
    #[validate(range(min = 1, max = 2))]
    #[builder(default = "None")]
    pub outer_dim: Option<usize>,

    /// Explicit inner dimension (head_size for MLA, num_heads * head_dim for standard).
    /// When set, skips shape inference. Python should always provide this.
    #[validate(range(min = 1))]
    #[builder(default = "None")]
    pub inner_dim: Option<usize>,

    #[builder(default = "None")]
    scheduler_client: Option<TransferSchedulerClient>,

    #[builder(default = "String::from(\"tcp://127.0.0.1:56001\")")]
    leader_pub_url: String,

    #[builder(default = "String::from(\"tcp://127.0.0.1:56002\")")]
    leader_ack_url: String,

    /// Rank for replicated mode (None = sharded mode)
    #[builder(default = "None")]
    rank: Option<i32>,

    /// NCCL configuration for replicated mode
    #[builder(default = "transfer::NcclConfig::disabled()")]
    nccl_config: transfer::NcclConfig,

    /// Per-group layer geometry. Empty (the default) keeps the legacy
    /// single-geometry synthesis.
    ///
    /// Grouped configs are derived per group with the flavor
    /// `LayerSeparate { outer_contiguous: true }`; the derivation ignores
    /// (and warns about) a `device_layout_type` that differs from it, so
    /// grouped callers should not rely on that field.
    #[builder(default = "Vec::new()")]
    pub groups: Vec<LayerGroup>,

    /// Bound on layers kept active for save operations; will be consumed by
    /// the hybrid save trigger (staged; not yet wired). Only meaningful for
    /// grouped configs, where it is validated against the registered tensor
    /// count.
    #[builder(default = "None")]
    pub active_save_layers: Option<usize>,
}

impl KvbmWorkerConfig {
    pub fn builder() -> KvbmWorkerConfigBuilder {
        KvbmWorkerConfigBuilder::default()
    }

    /// Validate configuration contract before use.
    ///
    /// Field-level rules (`outer_dim`, `inner_dim`, `page_size`) are enforced via
    /// `#[validate]` attributes on the struct. This method additionally checks the
    /// cross-field coupling invariant: `outer_dim` and `inner_dim` must both be
    /// `Some` or both be `None`.
    pub fn validate(&self) -> anyhow::Result<()> {
        // Run derive-based field validators (#[validate] attributes).
        <Self as Validate>::validate(self)
            .map_err(|e| anyhow::anyhow!("KvbmWorkerConfig validation failed: {}", e))?;

        // Cross-field: outer_dim and inner_dim must be coupled (both Some or both None).
        match (self.outer_dim, self.inner_dim) {
            (Some(_), None) | (None, Some(_)) => {
                anyhow::bail!(
                    "outer_dim and inner_dim must be provided together (both Some or both None); \
                     got outer_dim={:?}, inner_dim={:?}",
                    self.outer_dim,
                    self.inner_dim
                );
            }
            _ => {}
        }

        // Grouped configs: each group's `tensor_indices` must be strictly
        // increasing WITHIN the group, and across groups every registered
        // tensor must be claimed by exactly one group (union, no
        // duplicates, no gaps). Cross-group ordering is irrelevant — each
        // group's layout consumes only its own tensors — so interleaved
        // groups (hybrid attention layers inside a mamba span) are valid.
        // `active_save_layers` is bounded by the number of registered
        // tensors. Ungrouped (legacy) configs are not subject to the bound.
        if !self.groups.is_empty() {
            for (group_idx, group) in self.groups.iter().enumerate() {
                if group.tensor_indices.is_empty() {
                    anyhow::bail!(
                        "LayerGroup {} has no tensor indices; every group must cover at least one registered tensor",
                        group_idx
                    );
                }
                // A zero factor would make bytes_per_block() (and the
                // downstream region sizing) silently collapse to 0.
                if group.bytes_per_block() == 0 {
                    anyhow::bail!(
                        "LayerGroup {} outer_dim/inner_dim/page_bytes must be non-zero; got outer_dim={}, inner_dim={}, page_bytes={}",
                        group_idx,
                        group.outer_dim,
                        group.inner_dim,
                        group.page_bytes
                    );
                }
            }

            let covered: usize = self.groups.iter().map(|g| g.tensor_indices.len()).sum();
            if covered != self.tensors.len() {
                anyhow::bail!(
                    "LayerGroups must partition the registered tensors: groups cover {} tensor indices but {} tensors are registered",
                    covered,
                    self.tensors.len()
                );
            }

            // Exact partition: with the covered-count check above, having
            // as many distinct in-range claims as registered tensors
            // means every index is claimed exactly once. Within a group
            // the indices must be strictly increasing, i.e. registration
            // order; across groups any order is fine.
            let mut claimed_by = vec![Option::<(usize, usize)>::None; self.tensors.len()];
            for (group_idx, group) in self.groups.iter().enumerate() {
                for (pos, &tensor_idx) in group.tensor_indices.iter().enumerate() {
                    if tensor_idx >= self.tensors.len() {
                        anyhow::bail!(
                            "LayerGroup {} references tensor index {} but only {} tensors are registered",
                            group_idx,
                            tensor_idx,
                            self.tensors.len()
                        );
                    }
                    if let Some((prev_group, prev_pos)) = claimed_by[tensor_idx] {
                        anyhow::bail!(
                            "LayerGroup {} claims tensor index {} which is already claimed (by group {} at position {}); groups must partition the registered tensors with no duplicates",
                            group_idx,
                            tensor_idx,
                            prev_group,
                            prev_pos
                        );
                    }
                    if pos > 0 && tensor_idx <= group.tensor_indices[pos - 1] {
                        anyhow::bail!(
                            "LayerGroup {} tensor indices must be strictly increasing within the group; got {} after {}",
                            group_idx,
                            tensor_idx,
                            group.tensor_indices[pos - 1]
                        );
                    }
                    claimed_by[tensor_idx] = Some((group_idx, pos));
                }
            }

            if let Some(active_save_layers) = self.active_save_layers
                && active_save_layers > self.tensors.len()
            {
                anyhow::bail!(
                    "active_save_layers ({}) exceeds the number of registered tensors ({})",
                    active_save_layers,
                    self.tensors.len()
                );
            }
        }

        Ok(())
    }
}

/// Everything `worker_task` needs, bundled by value: the per-group layout
/// specs, the handshake-reported geometry, the worker config, and the cells
/// it publishes through.
struct WorkerTaskArgs {
    group_layouts: Vec<GroupLayoutSpec>,
    geometry: HandshakeGeometry,
    config: KvbmWorkerConfig,
    cancel_token: CancellationToken,
    handler_tx: Arc<Mutex<Option<oneshot::Sender<BlockTransferHandler>>>>,
    layout_ready_tx: tokio::sync::Mutex<Option<oneshot::Sender<String>>>,
    scheduler_client: Option<TransferSchedulerClient>,
}

pub struct KvbmWorker {
    task: Option<CriticalTaskExecutionHandle>,
    block_transfer_handler_rx: Option<oneshot::Receiver<transfer::BlockTransferHandler>>,
}

impl KvbmWorker {
    pub async fn new(config: KvbmWorkerConfig, layout_blocking: bool) -> anyhow::Result<Self> {
        tracing::info!(
            "Initializing KvbmWorker with params: num_device_blocks={}, page_size={}, dtype_width_bytes={}",
            config.num_device_blocks,
            config.page_size,
            config.dtype_width_bytes
        );

        config.validate()?;

        if config.num_device_blocks == 0 {
            return Err(anyhow::anyhow!("num_device_blocks must be greater than 0"));
        }

        let (device_tensors, shape) =
            load_and_validate_tensors(&config.tensors, config.device_id, &config.groups)?;

        if config.groups.is_empty() {
            if shape.len() < 3 {
                return Err(anyhow::anyhow!(format!(
                    "Unsupported kv cache layout. Got shape: {:?}",
                    shape
                )));
            }
        } else {
            // Grouped: validate per group. Attention groups keep the
            // >=3-dim layout contract; a mamba group's dense [B, C] rows
            // are legitimately 2D, so a mamba-first first-tensor must not
            // trip the legacy guard.
            for (group_idx, group) in config.groups.iter().enumerate() {
                if group.is_mamba() {
                    continue;
                }
                let tensor_idx = group.tensor_indices[0];
                let tensor_shape = config.tensors[tensor_idx].shape();
                if tensor_shape.len() < 3 {
                    return Err(anyhow::anyhow!(
                        "Unsupported kv cache layout for attention layer group {}: got shape {:?} (needs >= 3 dims)",
                        group_idx,
                        tensor_shape
                    ));
                }
            }
        }

        let (group_layouts, bytes_per_block, bytes_per_block_per_group) = if config
            .groups
            .is_empty()
        {
            let (layout_type, num_layers, outer_dim, inner_dim) = match config.device_layout_type {
                LayoutType::FullyContiguous => {
                    let num_layers = shape[1];
                    let outer_dim = config.outer_dim.unwrap_or(shape[2]);
                    let inner_dim = config
                        .inner_dim
                        .unwrap_or_else(|| shape[3..].iter().product::<usize>() / config.page_size);
                    tracing::info!(
                        "Inferred layout: num_layers={}, outer_dim={}, page_size={}, inner_dim={}",
                        num_layers,
                        outer_dim,
                        config.page_size,
                        inner_dim
                    );

                    (
                        LayoutType::FullyContiguous,
                        num_layers,
                        outer_dim,
                        inner_dim,
                    )
                }
                LayoutType::LayerSeparate { outer_contiguous } => {
                    let layout_type = config.device_layout_type;
                    let num_layers = device_tensors.len();

                    let (outer_dim, inner_dim) = infer_layer_separate_dims(
                        &shape,
                        config.page_size,
                        outer_contiguous,
                        (config.outer_dim, config.inner_dim),
                    );

                    tracing::info!(
                        "Layout: num_layers={}, outer_dim={}, outer_contiguous={}, page_size={}, inner_dim={}",
                        num_layers,
                        outer_dim,
                        outer_contiguous,
                        config.page_size,
                        inner_dim
                    );

                    (layout_type, num_layers, outer_dim, inner_dim)
                }
            };

            // Legacy scalar: the single synthesized group carries exactly
            // the product of the historical formula (KEY INVARIANT: an
            // empty `groups` reproduces today's geometry).
            let group = LayerGroup::from_legacy(
                num_layers,
                config.page_size,
                outer_dim,
                inner_dim,
                config.dtype_width_bytes,
                device_tensors.len(),
            );
            let bytes_per_block = group.bytes_per_block();

            let mut layout_builder_instance = LayoutConfigBuilder::default();
            let layout_builder = layout_builder_instance
                .num_layers(num_layers)
                .outer_dim(outer_dim)
                .page_size(config.page_size)
                .inner_dim(inner_dim)
                .dtype_width_bytes(config.dtype_width_bytes);

            let device_layout = layout_builder
                .num_blocks(config.num_device_blocks)
                .build()?
                .create_layout(layout_type, device_tensors)
                .map_err(|e| anyhow::anyhow!("legacy device layout: {e}"))?;

            (
                vec![GroupLayoutSpec {
                    geometry: GroupGeometry {
                        layout_type,
                        num_layers,
                        outer_dim,
                        inner_dim,
                        page_size: config.page_size,
                        dtype_width_bytes: config.dtype_width_bytes,
                        bytes_per_block,
                    },
                    device_layout,
                }],
                bytes_per_block,
                // Modern single-group worker: one group, one vec entry.
                vec![bytes_per_block],
            )
        } else {
            // Grouped path: derive each group's geometry — mamba groups
            // skip shape inference, attention groups infer from their
            // first registered tensor — and sum the per-block bytes.
            //
            // The grouped derivation only supports the per-layer flavor the
            // groups carry (LayerSeparate { outer_contiguous: true }). The
            // builder default is FullyContiguous and grouped callers are not
            // expected to set device_layout_type explicitly, so a differing
            // value is warned about and ignored rather than rejected.
            if config.device_layout_type
                != (LayoutType::LayerSeparate {
                    outer_contiguous: true,
                })
            {
                tracing::warn!(
                    "Grouped config has device_layout_type={:?}; the grouped derivation only supports LayerSeparate {{ outer_contiguous: true }} and will proceed with the groups' layout type",
                    config.device_layout_type
                );
            }
            let geometries: Vec<GroupGeometry> = config
                .groups
                .iter()
                .map(|group| {
                    derive_group_geometry(
                        group,
                        &config.tensors,
                        (config.outer_dim, config.inner_dim),
                        config.num_device_blocks,
                    )
                })
                .collect();
            let bytes_per_block = geometries
                .iter()
                .map(|geometry| geometry.bytes_per_block)
                .sum();
            let bytes_per_block_per_group: Vec<usize> = geometries
                .iter()
                .map(|geometry| geometry.bytes_per_block)
                .collect();

            // Per-group DEVICE layouts: each group's layout is built over
            // its OWN tensor subset (the groups partition the
            // registration-order tensor list — validate() enforces it),
            // with the group's derived geometry.
            let mut remaining: Vec<Option<DeviceStorage>> =
                device_tensors.into_iter().map(Some).collect();
            let mut group_layouts = Vec::with_capacity(geometries.len());
            for (group_idx, (group, geometry)) in config.groups.iter().zip(&geometries).enumerate()
            {
                let mut subset = Vec::with_capacity(group.tensor_indices.len());
                for &i in &group.tensor_indices {
                    // Defense-in-depth: `validate()` already enforces the
                    // exact partition this checks (each registered index
                    // is claimed by exactly one group), so the missing-arm
                    // case is unreachable in practice.
                    let Some(storage) = remaining[i].take() else {
                        return Err(anyhow::anyhow!(
                            "layer group {group_idx}: tensor index {i} already consumed by an earlier group (groups must partition the tensor list)"
                        ));
                    };
                    subset.push(storage);
                }
                let mut builder = LayoutConfigBuilder::default();
                let builder = builder
                    .num_layers(geometry.num_layers)
                    .outer_dim(geometry.outer_dim)
                    .page_size(geometry.page_size)
                    .inner_dim(geometry.inner_dim)
                    .dtype_width_bytes(geometry.dtype_width_bytes);
                let device_layout = builder
                    .num_blocks(config.num_device_blocks)
                    .build()?
                    .create_layout(geometry.layout_type, subset)
                    .map_err(|e| {
                        anyhow::anyhow!("device layout for layer group {group_idx}: {e}")
                    })?;
                group_layouts.push(GroupLayoutSpec {
                    geometry: *geometry,
                    device_layout,
                });
            }

            (group_layouts, bytes_per_block, bytes_per_block_per_group)
        };

        let geometry = HandshakeGeometry {
            bytes_per_block,
            bytes_per_block_per_group,
        };

        let (task, handler_rx) = if layout_blocking {
            Self::run_blocking_layout_initialization(config, group_layouts, geometry).await?
        } else {
            Self::run_non_blocking_layout_initialization(config, group_layouts, geometry).await?
        };

        Ok(Self {
            task: Some(task),
            block_transfer_handler_rx: Some(handler_rx),
        })
    }

    async fn run_blocking_layout_initialization(
        config: KvbmWorkerConfig,
        group_layouts: Vec<GroupLayoutSpec>,
        geometry: HandshakeGeometry,
    ) -> anyhow::Result<(
        CriticalTaskExecutionHandle,
        oneshot::Receiver<transfer::BlockTransferHandler>,
    )> {
        let cancel_token = config.cancel_token.clone();

        // establish a oneshot channel to get back the raw BlockTransferHandler
        let (handler_tx, handler_rx) = oneshot::channel();
        let handler_tx_cell = Arc::new(Mutex::new(Some(handler_tx)));

        // establish a oneshot channel to block on the main routine to wait for layout allocation readiness
        let (layout_ready_tx, layout_ready_rx) = oneshot::channel::<String>();
        let layout_ready_tx_cell = Mutex::new(Some(layout_ready_tx));

        let scheduler_client = config.scheduler_client.clone();

        // Move (not clone) — `config` has no further readers here.
        let worker_config = config;
        // start background worker task to do layout allocation for host or disk
        let task = CriticalTaskExecutionHandle::new(
            move |cancel_token| {
                KvbmWorker::worker_task(WorkerTaskArgs {
                    group_layouts,
                    geometry,
                    config: worker_config,
                    cancel_token,
                    handler_tx: handler_tx_cell,
                    layout_ready_tx: layout_ready_tx_cell,
                    scheduler_client,
                })
            },
            cancel_token.clone(),
            "kvbm-worker-task",
        )?;

        // waiting for the worker layout allocation ready
        match layout_ready_rx.await {
            Ok(_) => tracing::info!("worker layout allocation finished."),
            Err(_) => tracing::error!("Worker layout dropped without sending"),
        }

        Ok((task, handler_rx))
    }

    async fn run_non_blocking_layout_initialization(
        config: KvbmWorkerConfig,
        group_layouts: Vec<GroupLayoutSpec>,
        geometry: HandshakeGeometry,
    ) -> anyhow::Result<(
        CriticalTaskExecutionHandle,
        oneshot::Receiver<transfer::BlockTransferHandler>,
    )> {
        let cancel_token = config.cancel_token.clone();
        let scheduler_client = config.scheduler_client.clone();

        // channel to get BlockTransferHandler back to the caller
        let (handler_tx, handler_rx) = oneshot::channel::<transfer::BlockTransferHandler>();
        let handler_tx_cell = Arc::new(Mutex::new(Some(handler_tx)));

        // channel that the worker will use to signal layout readiness
        let (layout_ready_tx, layout_ready_rx) = oneshot::channel::<String>();
        let layout_ready_tx_cell = Mutex::new(Some(layout_ready_tx));

        // clone what we need inside the orchestrator
        let worker_config = config.clone();
        let cancel_token_for_task = cancel_token.clone();

        // Single task that orchestrates everything in-order.
        let task = CriticalTaskExecutionHandle::new(
            move |ct| {
                let args = WorkerTaskArgs {
                    group_layouts,
                    geometry,
                    config: worker_config.clone(),
                    cancel_token: ct.clone(),
                    handler_tx: handler_tx_cell,
                    layout_ready_tx: layout_ready_tx_cell,
                    scheduler_client: scheduler_client.clone(),
                };

                async move {
                    // Start the long-running worker.
                    let worker_fut = KvbmWorker::worker_task(args);

                    // If worker_task returns Result, handle/log it inside the spawned task.
                    tokio::spawn(async move {
                        if let Err(e) = worker_fut.await {
                            tracing::error!("worker_task exited with error: {e:#}");
                        }
                    });

                    // 3) wait for the worker’s layout allocation readiness
                    match layout_ready_rx.await {
                        Ok(_) => tracing::info!("worker layout allocation finished."),
                        Err(_) => tracing::warn!("worker layout readiness channel dropped"),
                    }

                    Ok::<(), anyhow::Error>(())
                }
            },
            cancel_token_for_task,
            "kvbm-worker-task",
        )?;

        Ok((task, handler_rx))
    }

    /// One-time use method to extract the block transfer handler from the worker.
    ///
    /// This is a bit of a hack. Improve the API design around this in the future.
    pub fn block_transfer_handler_rx(
        &mut self,
    ) -> Option<tokio::sync::oneshot::Receiver<BlockTransferHandler>> {
        self.block_transfer_handler_rx.take()
    }

    fn make_layout<S: Storage, M: BlockMetadata>(
        mut layout: Box<dyn NixlLayout<StorageType = S>>,
        agent: &Option<NixlAgent>,
        block_set_idx: usize,
        worker_id: usize,
    ) -> anyhow::Result<Vec<Block<S, locality::Local, M>>> {
        // Register with NIXL, if applicable.
        if let Some(agent) = agent {
            layout.nixl_register(agent, None)?;
        }

        // Convert the layout into blocks.
        let layout: Arc<dyn NixlLayout<StorageType = S>> = Arc::from(layout);
        let blocks = layout_to_blocks::<_, M>(layout, block_set_idx, worker_id as u64)?;
        Ok(blocks)
    }

    async fn worker_task(
        WorkerTaskArgs {
            group_layouts,
            geometry,
            config,
            cancel_token,
            handler_tx,
            layout_ready_tx,
            scheduler_client,
        }: WorkerTaskArgs,
    ) -> anyhow::Result<()> {
        let worker_id = config.device_id;
        // Readiness gating for ping
        let state = Arc::new(WorkerState::new());

        // Cell to publish the transfer handler
        let transfer_handler_cell: Arc<RwLock<Option<BlockTransferHandler>>> =
            Arc::new(RwLock::new(None));

        // Build handlers map
        let mut handlers: HashMap<String, Arc<dyn Handler>> = HashMap::new();

        handlers.insert(
            ZMQ_PING_MESSAGE.to_string(),
            Arc::new(GatedPing {
                state: state.clone(),
                layout_ready_tx,
            }) as Arc<dyn Handler>,
        );

        handlers.insert(
            ZMQ_WORKER_METADATA_MESSAGE.to_string(),
            Arc::new(WorkerMetadataHandler {
                num_device_blocks: config.num_device_blocks,
                bytes_per_block: geometry.bytes_per_block,
                bytes_per_block_per_group: geometry.bytes_per_block_per_group,
            }) as Arc<dyn Handler>,
        );

        handlers.insert(
            ZMQ_LEADER_METADATA_MESSAGE.to_string(),
            Arc::new(LeaderMetadataHandler {
                state: state.clone(),
                group_layouts: tokio::sync::Mutex::new(Some(group_layouts)), // moved in
                worker_config: config.clone(),
                worker_id,
                device_id: config.device_id,
                scheduler_client,
                handler_cell: transfer_handler_cell.clone(),
                handler_tx, // sends BlockTransferHandler to caller
                started: AtomicBool::new(false),
            }) as Arc<dyn Handler>,
        );

        // transfer requests get dispatched to built handler (after allocation)
        handlers.insert(
            ZMQ_TRANSFER_BLOCKS_MESSAGE.to_string(),
            Arc::new(BlockTransferDispatch {
                cell: transfer_handler_cell.clone(),
            }) as Arc<dyn Handler>,
        );

        let _zmq_worker = ZmqActiveMessageWorker::new(
            &config.leader_pub_url,
            &config.leader_ack_url,
            handlers,
            cancel_token.clone(),
        )?;

        // TODO: Some sort of fancy loop here.
        // For now, just wait for cancellation.
        cancel_token.cancelled().await;

        Ok(())
    }
}

impl Drop for KvbmWorker {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.cancel();
            task.detach();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    fn base_config() -> KvbmWorkerConfig {
        KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(1)
            .build()
            .expect("base config should build")
    }

    // --- outer_dim ---

    #[test]
    fn validate_outer_dim_none_is_ok() {
        let mut cfg = base_config();
        cfg.outer_dim = None;
        cfg.inner_dim = None;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_outer_dim_1_is_ok() {
        let mut cfg = base_config();
        cfg.outer_dim = Some(1);
        cfg.inner_dim = Some(64);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_outer_dim_2_is_ok() {
        let mut cfg = base_config();
        cfg.outer_dim = Some(2);
        cfg.inner_dim = Some(64);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_outer_dim_3_is_err() {
        let mut cfg = base_config();
        cfg.outer_dim = Some(3);
        cfg.inner_dim = Some(64);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("outer_dim") && err.contains("range"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_outer_dim_0_is_err() {
        let mut cfg = base_config();
        cfg.outer_dim = Some(0);
        cfg.inner_dim = Some(64);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("outer_dim") && err.contains("range"),
            "got: {err}"
        );
    }

    // --- inner_dim ---

    #[test]
    fn validate_inner_dim_zero_is_err() {
        let mut cfg = base_config();
        cfg.outer_dim = Some(2);
        cfg.inner_dim = Some(0);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("inner_dim") && err.contains("range"),
            "got: {err}"
        );
    }

    // --- coupling ---

    #[test]
    fn validate_outer_some_inner_none_is_err() {
        let mut cfg = base_config();
        cfg.outer_dim = Some(2);
        cfg.inner_dim = None;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("provided together"), "got: {err}");
    }

    #[test]
    fn validate_outer_none_inner_some_is_err() {
        let mut cfg = base_config();
        cfg.outer_dim = None;
        cfg.inner_dim = Some(64);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("provided together"), "got: {err}");
    }

    // --- page_size ---

    #[test]
    fn validate_page_size_power_of_two_is_ok() {
        for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let mut cfg = base_config();
            cfg.page_size = size;
            assert!(cfg.validate().is_ok(), "expected ok for page_size={size}");
        }
    }

    #[test]
    fn validate_page_size_not_power_of_two_is_err() {
        for size in [3, 5, 6, 7, 100, 300] {
            let mut cfg = base_config();
            cfg.page_size = size;
            let err = cfg.validate().unwrap_err().to_string();
            assert!(
                err.contains("page_size_not_power_of_two"),
                "expected power-of-two error for page_size={size}, got: {err}"
            );
        }
    }

    #[test]
    fn validate_page_size_exceeds_max_is_err() {
        let mut cfg = base_config();
        cfg.page_size = 2048;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("page_size") && err.contains("range"),
            "got: {err}"
        );
    }
}

#[cfg(test)]
mod layer_group_tests {
    use super::*;
    use crate::block_manager::storage::torch::TorchDevice;

    /// Minimal tensor stand-in for config-level validation (no CUDA needed).
    #[derive(Debug)]
    struct FakeTensor {
        shape: Vec<usize>,
    }

    impl TorchTensor for FakeTensor {
        fn device(&self) -> TorchDevice {
            TorchDevice::Other("fake".to_string())
        }

        fn data_ptr(&self) -> u64 {
            0
        }

        fn size_bytes(&self) -> usize {
            self.shape.iter().product()
        }

        fn shape(&self) -> Vec<usize> {
            self.shape.clone()
        }

        fn stride(&self) -> Vec<usize> {
            let mut stride = vec![1usize];
            for i in (0..self.shape.len().saturating_sub(1)).rev() {
                stride.push(stride.last().unwrap() * self.shape[i]);
            }
            stride.reverse();
            stride
        }
    }

    #[test]
    fn group_geometry_mamba_and_attention() {
        // attention group: 4 layers, K/V outer=2, page 784 tokens, inner 512,
        // fp8 (dtype factor 1, folded)
        let attn = LayerGroup::attention(4, 784, 2, 512, 1);
        assert_eq!(attn.bytes_per_block(), 4 * 2 * 784 * 512);
        // mamba group: 24 layers, dense int8 [B, C] rows, C = 3_207_168
        let mamba = LayerGroup::mamba(24, 3_207_168);
        assert_eq!(mamba.bytes_per_block(), 24 * 3_207_168);
        assert!(mamba.is_mamba());
        assert_eq!(
            mamba.layout_type(),
            LayoutType::LayerSeparate {
                outer_contiguous: true
            }
        );
    }

    #[test]
    fn total_bytes_per_block_sums_groups() {
        let groups = [
            LayerGroup::attention(4, 784, 2, 512, 1),
            LayerGroup::mamba(24, 3_207_168),
        ];
        let total: usize = groups.iter().map(|g| g.bytes_per_block()).sum();
        assert_eq!(total, 4 * 2 * 784 * 512 + 24 * 3_207_168);
    }

    /// THE KEY INVARIANT: with `groups` empty (every caller today), the
    /// synthesized single group must reproduce the legacy geometry EXACTLY.
    /// The legacy scalar is recomputed here independently of
    /// `bytes_per_block`, so the check is not a tautology.
    #[test]
    fn legacy_synthesis_reproduces_legacy_geometry() {
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(64)
            .page_size(784)
            .dtype_width_bytes(1)
            .build()
            .expect("legacy config should build");
        assert!(config.groups.is_empty());
        assert!(config.active_save_layers.is_none());
        assert!(config.validate().is_ok());

        // Scalar geometry the legacy LayerSeparate inference derives for a
        // standard K/V model: 28 layer tensors, outer=2, page 784, inner 512.
        let (num_layers, outer_dim, inner_dim, num_tensors) = (28usize, 2, 512, 28);
        let group = LayerGroup::from_legacy(
            num_layers,
            config.page_size,
            outer_dim,
            inner_dim,
            config.dtype_width_bytes,
            num_tensors,
        );

        // Legacy scalar computation, spelled out independently.
        let legacy_bytes_per_block =
            num_layers * outer_dim * config.page_size * inner_dim * config.dtype_width_bytes;
        assert_eq!(group.bytes_per_block(), legacy_bytes_per_block);
        assert_eq!(group.num_layers, num_layers);
        assert_eq!(group.page_bytes, config.page_size);
        assert_eq!(group.outer_dim, outer_dim);
        assert_eq!(group.inner_dim, inner_dim);
        assert_eq!(group.dtype_width_bytes, config.dtype_width_bytes);
        assert!(!group.is_mamba());
        assert_eq!(
            group.tensor_indices,
            (0..num_tensors).collect::<Vec<usize>>()
        );

        // FullyContiguous legacy: num_layers comes from shape[1] and may
        // differ from the tensor count (one tensor holding every layer).
        let fc_group = LayerGroup::from_legacy(80, 32, 2, 576, 2, 1);
        assert_eq!(fc_group.bytes_per_block(), 80 * 2 * 32 * 576 * 2);
        assert_eq!(fc_group.tensor_indices, vec![0]);

        // active_save_layers is only bounded when groups are present.
        let mut no_groups = config.clone();
        no_groups.active_save_layers = Some(9999);
        assert!(no_groups.validate().is_ok());
    }

    #[test]
    fn validate_groups_partition_happy_path() {
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
            Arc::new(FakeTensor {
                shape: vec![8, 128],
            }),
        ];
        let mut attn = LayerGroup::attention(2, 16, 2, 64, 2);
        attn.tensor_indices = vec![0, 1];
        let mut mamba = LayerGroup::mamba(1, 128);
        mamba.tensor_indices = vec![2];
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .tensors(tensors)
            .groups(vec![attn, mamba])
            .build()
            .expect("grouped config should build");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_groups_accept_interleaved_hma_partition() {
        // The hybrid fp4 pattern: attention layers at positions 3 and 6
        // inside the mamba span. The groups' concatenation
        // ([0,1,2,4,5] ++ [3,6]) is NOT monotonic, but each group's own
        // indices are strictly increasing and together they claim every
        // registered tensor exactly once — a valid partition.
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(FakeTensor {
                shape: vec![8, 128],
            }), // 0 mamba
            Arc::new(FakeTensor {
                shape: vec![8, 128],
            }), // 1 mamba
            Arc::new(FakeTensor {
                shape: vec![8, 128],
            }), // 2 mamba
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }), // 3 attention
            Arc::new(FakeTensor {
                shape: vec![8, 128],
            }), // 4 mamba
            Arc::new(FakeTensor {
                shape: vec![8, 128],
            }), // 5 mamba
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }), // 6 attention
        ];
        let mut mamba = LayerGroup::mamba(5, 128);
        mamba.tensor_indices = vec![0, 1, 2, 4, 5];
        let mut attn = LayerGroup::attention(2, 16, 2, 64, 2);
        attn.tensor_indices = vec![3, 6];

        // Mamba-first group order (the fp4's actual registration layout).
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .tensors(tensors.clone())
            .groups(vec![mamba.clone(), attn.clone()])
            .active_save_layers(Some(2))
            .build()
            .expect("grouped config should build");
        assert!(config.validate().is_ok());

        // Cross-group order is irrelevant: the attention group may also
        // come first.
        let reordered = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .tensors(tensors)
            .groups(vec![attn, mamba])
            .active_save_layers(Some(2))
            .build()
            .expect("grouped config should build");
        assert!(reordered.validate().is_ok());
    }

    #[test]
    fn validate_groups_reject_within_group_disorder() {
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
        ];
        let mut attn = LayerGroup::attention(2, 16, 2, 64, 2);
        attn.tensor_indices = vec![1, 0];
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .tensors(tensors)
            .groups(vec![attn])
            .build()
            .expect("config should build");
        let err = config.validate().unwrap_err().to_string();
        assert!(
            err.contains("strictly increasing within the group"),
            "expected within-group order error, got: {err}"
        );
    }

    #[test]
    fn validate_groups_reject_duplicate_index_across_groups() {
        // Both groups claim tensor 0; the covered-count check passes (3
        // claims over 3 tensors) so the duplicate itself must be caught,
        // with the message naming the earlier claim's group and position.
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
        ];
        let mut first = LayerGroup::attention(2, 16, 2, 64, 2);
        first.tensor_indices = vec![0, 1];
        let mut second = LayerGroup::attention(1, 16, 2, 64, 2);
        second.tensor_indices = vec![0];
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .tensors(tensors)
            .groups(vec![first, second])
            .build()
            .expect("config should build");
        let err = config.validate().unwrap_err().to_string();
        assert!(
            err.contains("already claimed (by group 0 at position 0)"),
            "expected duplicate-claim error naming the earlier claim, got: {err}"
        );
    }

    #[test]
    fn validate_groups_reject_gap_in_coverage() {
        // No group claims tensor 2: the covered-count check (2 claims
        // over 3 tensors) reports the broken partition.
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
        ];
        let mut attn = LayerGroup::attention(1, 16, 2, 64, 2);
        attn.tensor_indices = vec![0];
        let mut mamba = LayerGroup::mamba(1, 128);
        mamba.tensor_indices = vec![1];
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .tensors(tensors)
            .groups(vec![attn, mamba])
            .build()
            .expect("config should build");
        let err = config.validate().unwrap_err().to_string();
        assert!(
            err.contains("partition"),
            "expected partition error, got: {err}"
        );
    }

    #[test]
    fn validate_groups_reject_out_of_range_index() {
        // Covered-count matches (1 claim over 1 tensor), but the claimed
        // index is beyond the registered tensor list.
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![Arc::new(FakeTensor {
            shape: vec![2, 4, 16, 64],
        })];
        let mut attn = LayerGroup::attention(1, 16, 2, 64, 2);
        attn.tensor_indices = vec![7];
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .tensors(tensors)
            .groups(vec![attn])
            .build()
            .expect("config should build");
        let err = config.validate().unwrap_err().to_string();
        assert!(
            err.contains("but only 1 tensors are registered"),
            "expected out-of-range error, got: {err}"
        );
    }

    #[test]
    fn validate_groups_reject_partition_size_mismatch() {
        let mut attn = LayerGroup::attention(1, 16, 2, 64, 2);
        attn.tensor_indices = vec![0];
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .groups(vec![attn])
            .build()
            .expect("config should build");
        let err = config.validate().unwrap_err().to_string();
        assert!(
            err.contains("partition"),
            "expected partition-size error, got: {err}"
        );
    }

    #[test]
    fn validate_groups_reject_empty_group() {
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .groups(vec![LayerGroup::attention(1, 16, 2, 64, 2)])
            .build()
            .expect("config should build");
        let err = config.validate().unwrap_err().to_string();
        assert!(
            err.contains("no tensor indices"),
            "expected empty-group error, got: {err}"
        );
    }

    #[test]
    fn validate_active_save_layers_bounded_when_grouped() {
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
        ];
        let mut attn = LayerGroup::attention(2, 16, 2, 64, 2);
        attn.tensor_indices = vec![0, 1];

        let over = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .tensors(tensors.clone())
            .groups(vec![attn.clone()])
            .active_save_layers(Some(3))
            .build()
            .expect("config should build");
        let err = over.validate().unwrap_err().to_string();
        assert!(
            err.contains("active_save_layers"),
            "expected active_save_layers error, got: {err}"
        );

        let ok = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .tensors(tensors)
            .groups(vec![attn])
            .active_save_layers(Some(2))
            .build()
            .expect("config should build");
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn validate_groups_reject_zero_page_bytes() {
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
        ];
        let mut attn = LayerGroup::attention(2, 0, 2, 512, 1);
        attn.tensor_indices = vec![0, 1];
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(4)
            .tensors(tensors)
            .groups(vec![attn])
            .build()
            .expect("config should build");
        let err = config.validate().unwrap_err().to_string();
        assert!(
            err.contains("page_bytes"),
            "expected page_bytes error, got: {err}"
        );
    }

    #[test]
    fn derive_group_geometry_attention_scopes_inference_to_first_tensor() {
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
            Arc::new(FakeTensor {
                shape: vec![2, 4, 16, 64],
            }),
        ];
        let mut group = LayerGroup::attention(2, 16, 2, 64, 2);
        group.tensor_indices = vec![0, 1];
        // shape [2, 4, 16, 64] with num_blocks=4: shape[0]=2 < 4 ->
        // outer_contiguous=true (the K/V-outer convention).
        let geometry = derive_group_geometry(&group, &tensors, (None, None), 4);
        // outer_contiguous shape [outer, n_blocks, page, inner]:
        // candidate = 2 <= 2 -> outer=2, inner = (16*64)/16 = 64.
        assert_eq!(geometry.outer_dim, 2);
        assert_eq!(geometry.inner_dim, 64);
        assert_eq!(geometry.num_layers, 2);
        assert_eq!(geometry.page_size, 16);
        assert_eq!(geometry.dtype_width_bytes, 2);
        assert_eq!(geometry.bytes_per_block, 2 * 2 * 16 * 64 * 2);
        assert_eq!(
            geometry.layout_type,
            LayoutType::LayerSeparate {
                outer_contiguous: true
            }
        );
    }

    #[test]
    fn derive_group_geometry_mamba_passthrough() {
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![
            Arc::new(FakeTensor {
                shape: vec![8, 128],
            }),
            Arc::new(FakeTensor {
                shape: vec![8, 128],
            }),
        ];
        let mut group = LayerGroup::mamba(2, 128);
        group.tensor_indices = vec![0, 1];
        let geometry = derive_group_geometry(&group, &tensors, (None, None), 8);
        // Mamba geometry is authoritative from the group (1/1/C/1); the 2D
        // [B, C] tensor shape is never inferred over.
        assert_eq!(geometry.outer_dim, 1);
        assert_eq!(geometry.inner_dim, 1);
        assert_eq!(geometry.page_size, 128);
        assert_eq!(geometry.dtype_width_bytes, 1);
        assert_eq!(geometry.num_layers, 2);
        assert_eq!(geometry.bytes_per_block, 2 * 128);
    }

    #[test]
    fn lbnhc_live_shape_takes_the_mla_else_branch() {
        // vLLM 0.29's LBNHC cache: [num_blocks, kv_heads=8, block_size=16,
        // 2*head_size=128] (the live 0.6B). The inference must take the
        // MLA-style else-branch (candidate = kv_heads = 8 > 2) and derive
        // outer=1, inner = (8*16*128)/16 = 1024 — the geometry the pinned
        // byte-parity GPU test then exercises.
        let (outer, inner) = infer_layer_separate_dims(
            &[64, 8, 16, 128],
            16,
            false, // outer_contiguous: shape[0]=64 >= num_blocks -> false
            (None, None),
        );
        assert_eq!((outer, inner), (1, 1024));
        // And the per-layer region: page(16) x inner(1024) x dtype(2) =
        // 32,768 B == the true 8*16*128*2 block slice.
        assert_eq!(16 * inner * 2, 8 * 16 * 128 * 2);
    }

    #[test]
    fn derive_group_geometry_lbnhc_block_first_shape_derives_false_inner_1024() {
        // THE REGRESSION PIN (the 0.6B determinism root cause): an attention
        // group over vLLM 0.29 LBNHC tensors [nb, kv_heads, tokens, 2*head]
        // must detect the BLOCK-FIRST convention (shape[0]=nb >= nb ->
        // outer_contiguous=false) and derive inner from shape[1..]:
        // (8*16*128)/16 = 1024. The hard-coded true derived inner from
        // shape[2..] (128) — 1/8th the bytes — truncating every stored
        // block to 16,384 of its 131,072 bytes.
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![Arc::new(FakeTensor {
            shape: vec![64, 8, 16, 128],
        })];
        let mut group = LayerGroup::attention(1, 16, 1, 1024, 2);
        group.tensor_indices = vec![0];
        let geometry = derive_group_geometry(&group, &tensors, (None, None), 64);
        assert_eq!(
            geometry.layout_type,
            LayoutType::LayerSeparate {
                outer_contiguous: false
            },
            "LBNHC's block-first axis must detect outer_contiguous=false"
        );
        assert_eq!(geometry.outer_dim, 1);
        assert_eq!(
            geometry.inner_dim, 1024,
            "inner from shape[1..], not shape[2..]"
        );
        assert_eq!(geometry.bytes_per_block, 1 * 1 * 16 * 1024 * 2);
    }

    #[test]
    fn validate_page_size_function_semantics() {
        // Zero is never a valid page size.
        let err = validate_page_size(0).unwrap_err();
        assert!(
            err.to_string().contains("page_size_not_power_of_two"),
            "got: {err}"
        );
        // 16-byte-aligned pages are accepted (784 = 16 x 49).
        assert!(validate_page_size(784).is_ok());
        // Non-aligned, non-power-of-two sizes are rejected.
        assert!(validate_page_size(100).is_err());
        assert!(validate_page_size(1028).is_err());
        // Powers of two remain accepted (the >1024 cap lives on the field
        // attribute, not here).
        assert!(validate_page_size(512).is_ok());
        assert!(validate_page_size(1024).is_ok());
    }
}

/// GPU-gated two-group (hybrid attention/mamba) offload/onboard roundtrip
/// tests — the proof tests for the KVBM multi-group plan. Requires a real
/// CUDA device plus the `testing-cuda` feature, exactly like the devbar
/// roundtrip tests (61ffb0ab0 / 802e44153); a no-GPU machine fails at
/// cudarc/context setup, so the CPU suite never runs these.
///
/// What is proven, per executor path, at the pool/executor level (the full
/// connector + leader/worker handshake stays with the e2e):
/// - per-group uniform pools sharing one block index (`16+g` / `32+g` sets),
///   built through `perform_allocation_and_build_handler`'s exact wiring
///   (the real-path construction test calls that function itself),
/// - group-tagged `BlockTransferRequest`s moving ONLY the addressed group's
///   pools, with byte-equality per group (no cross-group bleed, no
///   uniform-size truncation — the two groups' per-block bytes differ by
///   one 512-B quantum on purpose),
/// - Task 4's allocator partitioning: per-group host windows carved
///   sequentially (attention region first, mamba second) out of ONE
///   `DevbarAllocator`, verified straight from a file-backed Bar region,
/// - Task 5 end-to-end: the per-geometry kernel-dims cache (pinned-host
///   custom-kernel run) and the per-layer staged segment plans (staged runs).
///
/// Host-tier semantics coverage: StagedCpu (the raw-MMIO default) against a
/// mock window and against a file-backed Bar region; HostPointer (pinned
/// fallback) through the custom kernel. DirectIo needs the IOMEMORY
/// registration — root plus a real BAR aperture (the existing direct-io
/// roundtrip test's guard, mirrored by `..._real_bar` below); there is no
/// mock-window io-mapping path (io mappings only exist when the
/// registration ran), so the node run-script skips that test by name.
///
/// The logical transfer counters are NOT observable at this layer — they
/// live in the slot/manager machinery above (`metrics_kvbm`), covered by
/// the manager-level e2e — so they are intentionally not asserted here.
///
/// The ENV_LOCK guard is deliberately held across `.await`s in the async
/// tests: `#[tokio::test]` runs a CURRENT-THREAD runtime, so no other task
/// can interleave while it is held (no deadlock, no lost wakeup), and the
/// devbar env must stay pinned for the whole construction-and-run window.
#[cfg(all(test, feature = "testing-cuda"))]
#[allow(clippy::await_holding_lock)]
mod two_group_tests {
    use super::*;
    use crate::block_manager::block::{
        BasicMetadata, BlockDataExt, BlockDataProvider, BlockDataProviderMut,
        data::local::LocalBlockData, transfer::HostTransferSemantics,
    };
    use crate::block_manager::storage::DevbarStorage;
    use crate::block_manager::storage::torch::TorchDevice;
    use cudarc::driver::sys;
    use dynamo_runtime::config::environment_names::kvbm::devbar as env_devbar;
    use std::any::Any;
    use tokio_util::sync::CancellationToken;

    // The fp4's real shape, scaled down to node-window-sized allocations
    // (the controller's GPU window is ~hundreds of MiB; the fp4's 3 MB
    // rows would be reckless here). Two fake layers per group, 64 blocks.
    //
    // Attention group: 2 layers of 4D [B, outer=2, heads=16, dim=64] fp16
    // tensors — page 64 tokens × inner 16 elements × 2 bytes × 2 outer =
    // 4,096 bytes per layer-block, 8,192 bytes per block.
    const NUM_DEVICE_BLOCKS: usize = 64;
    const ATTN_LAYERS: usize = 2;
    const ATTN_PAGE_TOKENS: usize = 64;
    const ATTN_OUTER: usize = 2;
    const ATTN_INNER: usize = 16;
    const ATTN_DTYPE_BYTES: usize = 2;
    const ATTN_LAYER_BLOCK_BYTES: usize =
        ATTN_OUTER * ATTN_PAGE_TOKENS * ATTN_INNER * ATTN_DTYPE_BYTES; // 4,096
    const ATTN_BYTES_PER_BLOCK: usize = ATTN_LAYERS * ATTN_LAYER_BLOCK_BYTES; // 8,192

    // Mamba group: 2 layers of 2D [B, C] uint8 rows with C exactly one
    // 512-B quantum below the attention per-layer bytes — the fp4's
    // 4096-quantum-off ratio (3,211,264 vs 3,207,168), scaled 8:1.
    const MAMBA_LAYERS: usize = 2;
    const MAMBA_C: usize = ATTN_LAYER_BLOCK_BYTES - 512; // 3,584 (16-aligned)
    const MAMBA_BYTES_PER_BLOCK: usize = MAMBA_LAYERS * MAMBA_C; // 7,168

    const BYTES_PER_BLOCK_PER_GROUP: [usize; 2] = [ATTN_BYTES_PER_BLOCK, MAMBA_BYTES_PER_BLOCK];

    const NUM_HOST_BLOCKS: usize = NUM_DEVICE_BLOCKS;
    /// The real-BAR direct-io run keeps its host windows small: the expander
    /// stand-in's aperture size is not guaranteed beyond the 64 KiB the
    /// existing direct-io roundtrip test carves.
    const NUM_HOST_BLOCKS_REAL_BAR: usize = 8;
    /// Backing file for the shared-aperture test: both groups' page-rounded
    /// windows plus a zero tail that must stay untouched.
    const APERTURE_FILE_BYTES: usize = 2 * 1024 * 1024;

    /// Block ids moved by every scenario: K, plus its neighbor to catch
    /// off-by-one indexing between the two groups' pools.
    const MOVED_BLOCK_IDS: [usize; 2] = [5, 6];

    /// Serializes this module's devbar env mutations and the reads they can
    /// race (mirrors the devbar tests' ENV_LOCK; the two locks cannot see
    /// each other, so a full parallel `testing-cuda` suite may still flake
    /// across modules — the canonical invocation filters `two_group`).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A fake per-layer CUDA tensor: the e2e MockTensor pattern, made
    /// dtype-aware. The device allocation is `ManuallyDrop`'d — the CUDA
    /// memory lives until process exit, exactly like the existing e2e's
    /// tensors.
    #[derive(Debug, Clone)]
    struct FakeLayerTensor {
        ptr: u64,
        size: usize,
        shape: Vec<usize>,
    }

    impl FakeLayerTensor {
        fn new(shape: Vec<usize>, elem_width_bytes: usize) -> Self {
            let allocator = DeviceAllocator::new(0).expect("cuda device 0");
            let size = shape.iter().product::<usize>() * elem_width_bytes;
            let device_storage = std::mem::ManuallyDrop::new(
                allocator
                    .allocate(size)
                    .expect("fake layer tensor allocation"),
            );
            Self {
                ptr: device_storage.addr(),
                size,
                shape,
            }
        }
    }

    impl TorchTensor for FakeLayerTensor {
        fn device(&self) -> TorchDevice {
            TorchDevice::Cuda(0)
        }

        fn data_ptr(&self) -> u64 {
            self.ptr
        }

        fn size_bytes(&self) -> usize {
            self.size
        }

        fn shape(&self) -> Vec<usize> {
            self.shape.clone()
        }

        fn stride(&self) -> Vec<usize> {
            // Contiguous strides.
            let mut stride = vec![1];
            for i in (0..self.shape.len() - 1).rev() {
                stride.push(stride.last().unwrap() * self.shape[i]);
            }
            stride.reverse();
            stride
        }
    }

    /// Registration order [attn0, attn1, mamba0, mamba1]; the groups
    /// partition it as [0,1] and [2,3]. The attention tensors are 4D
    /// [B, outer=2, heads=16, dim=64] fp16 with heads×dim = page×inner
    /// elements; the mamba tensors are 2D [B, C] uint8 rows.
    fn fake_layer_tensors() -> Vec<Arc<dyn TorchTensor>> {
        let attention_shape = vec![NUM_DEVICE_BLOCKS, ATTN_OUTER, 16, 64];
        let mamba_shape = vec![NUM_DEVICE_BLOCKS, MAMBA_C];
        vec![
            Arc::new(FakeLayerTensor::new(
                attention_shape.clone(),
                ATTN_DTYPE_BYTES,
            )),
            Arc::new(FakeLayerTensor::new(attention_shape, ATTN_DTYPE_BYTES)),
            Arc::new(FakeLayerTensor::new(mamba_shape.clone(), 1)),
            Arc::new(FakeLayerTensor::new(mamba_shape, 1)),
        ]
    }

    /// The 2-group KvbmWorkerConfig mirroring the fp4's hybrid shape:
    /// one attention group + one mamba group over the fake layer tensors.
    fn two_group_config() -> KvbmWorkerConfig {
        let mut attention = LayerGroup::attention(
            ATTN_LAYERS,
            ATTN_PAGE_TOKENS,
            ATTN_OUTER,
            ATTN_INNER,
            ATTN_DTYPE_BYTES,
        );
        attention.tensor_indices = vec![0, 1];
        let mut mamba = LayerGroup::mamba(MAMBA_LAYERS, MAMBA_C);
        mamba.tensor_indices = vec![2, 3];
        KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(NUM_DEVICE_BLOCKS)
            .tensors(fake_layer_tensors())
            .device_id(0)
            .outer_dim(Some(ATTN_OUTER))
            .inner_dim(Some(ATTN_INNER))
            .groups(vec![attention, mamba])
            .build()
            .expect("two-group config builds")
    }

    /// `KvbmWorker::new`'s grouped arm, verbatim: the >=3-dim attention
    /// guard, per-group geometry derivation, and per-group DEVICE layouts
    /// over each group's own tensor subset.
    fn build_two_group_specs(
        config: &KvbmWorkerConfig,
    ) -> anyhow::Result<(Vec<GroupLayoutSpec>, HandshakeGeometry)> {
        let (device_tensors, _shape) =
            load_and_validate_tensors(&config.tensors, config.device_id, &config.groups)?;

        for (group_idx, group) in config.groups.iter().enumerate() {
            if group.is_mamba() {
                continue;
            }
            let tensor_shape = config.tensors[group.tensor_indices[0]].shape();
            anyhow::ensure!(
                tensor_shape.len() >= 3,
                "Unsupported kv cache layout for attention layer group {}: got shape {:?} (needs >= 3 dims)",
                group_idx,
                tensor_shape
            );
        }

        let geometries: Vec<GroupGeometry> = config
            .groups
            .iter()
            .map(|group| {
                derive_group_geometry(
                    group,
                    &config.tensors,
                    (config.outer_dim, config.inner_dim),
                    config.num_device_blocks,
                )
            })
            .collect();
        let bytes_per_block = geometries
            .iter()
            .map(|geometry| geometry.bytes_per_block)
            .sum();
        let bytes_per_block_per_group: Vec<usize> = geometries
            .iter()
            .map(|geometry| geometry.bytes_per_block)
            .collect();

        let mut remaining: Vec<Option<DeviceStorage>> =
            device_tensors.into_iter().map(Some).collect();
        let mut group_layouts = Vec::with_capacity(geometries.len());
        for (group_idx, (group, geometry)) in config.groups.iter().zip(&geometries).enumerate() {
            let mut subset = Vec::with_capacity(group.tensor_indices.len());
            for &i in &group.tensor_indices {
                let Some(storage) = remaining[i].take() else {
                    anyhow::bail!(
                        "layer group {group_idx}: tensor index {i} already consumed by an earlier group (groups must partition the tensor list)"
                    );
                };
                subset.push(storage);
            }
            let mut builder = LayoutConfigBuilder::default();
            let builder = builder
                .num_layers(geometry.num_layers)
                .outer_dim(geometry.outer_dim)
                .page_size(geometry.page_size)
                .inner_dim(geometry.inner_dim)
                .dtype_width_bytes(geometry.dtype_width_bytes);
            let device_layout = builder
                .num_blocks(config.num_device_blocks)
                .build()?
                .create_layout(geometry.layout_type, subset)
                .map_err(|e| anyhow::anyhow!("device layout for layer group {group_idx}: {e}"))?;
            group_layouts.push(GroupLayoutSpec {
                geometry: *geometry,
                device_layout,
            });
        }

        Ok((
            group_layouts,
            HandshakeGeometry {
                bytes_per_block,
                bytes_per_block_per_group,
            },
        ))
    }

    /// Per-group pools + the handler driving them, plus retained views over
    /// the same pool-owned memory (the handler is opaque by design; these
    /// clones use the handler's own extraction so the test can stamp and
    /// verify what the executor moves).
    struct TwoGroupPools {
        handler: BlockTransferHandler,
        device_data: Vec<Vec<LocalBlockData<DeviceStorage>>>,
        host_data: Vec<Vec<LocalBlockData<DevbarStorage>>>,
        /// The per-window io mappings wired into the context (non-empty only
        /// when the backing is a registered BAR window).
        io_mappings: Vec<(usize, u64, usize)>,
    }

    type LocalBlockList<S> = Vec<Block<S, locality::Local, BasicMetadata>>;

    /// `perform_allocation_and_build_handler`'s wiring, verbatim except:
    /// the host allocator is injected (the real fn takes it from `from_env`),
    /// there is no disk tier, and `LocalBlockData` clones are retained so
    /// the test can observe pool contents. Per-group pools are allocated
    /// sequentially from the ONE shared allocator — the Bar carve order the
    /// shared-aperture test asserts.
    async fn build_two_group_pools(
        config: &KvbmWorkerConfig,
        host_allocator: Arc<DevbarAllocator>,
        semantics: HostTransferSemantics,
        num_host_blocks: usize,
    ) -> anyhow::Result<TwoGroupPools> {
        config.validate()?;
        let (group_layouts, _geometry) = build_two_group_specs(config)?;
        let num_groups = group_layouts.len();
        // Callers assert their own group count (the two-group scenarios
        // assert 2; the single-group LBNHC baseline asserts 1).

        let agent = build_agent(0, false)?;
        let pool_config = PoolConfig {
            enable_pool: true,
            max_concurrent_transfers: max_concurrent_transfers(),
            max_transfer_batch_size: max_transfer_batch_size(),
            num_outer_components: group_layouts
                .iter()
                .map(|spec| spec.geometry.outer_dim)
                .max()
                .unwrap_or(1),
            num_layers: group_layouts
                .iter()
                .map(|spec| spec.geometry.num_layers)
                .max()
                .unwrap_or(1),
        };
        // Bind the primary context before anything allocates pinned memory
        // (the PinnedStorage SAFETY contract) and before the test's own
        // cuMemcpy calls.
        DeviceAllocator::new(0)?.ctx().bind_to_thread()?;
        let mut transfer_context = TransferContext::new(
            Arc::new(Some(agent)),
            DeviceAllocator::new(0)?.ctx().new_stream()?,
            Handle::current(),
            Some(pool_config),
        )
        .map_err(|e| anyhow::anyhow!("transfer context: {e}"))?;
        transfer_context.set_host_transfer_semantics(semantics);
        let transfer_context = Arc::new(transfer_context);

        let mut device_blocks: Vec<Option<LocalBlockList<DeviceStorage>>> =
            Vec::with_capacity(num_groups);
        let mut host_blocks: Vec<Option<LocalBlockList<DevbarStorage>>> =
            Vec::with_capacity(num_groups);
        let mut host_io_mappings: Vec<(usize, u64, usize)> = Vec::new();

        for (group_idx, spec) in group_layouts.into_iter().enumerate() {
            let GroupLayoutSpec {
                geometry,
                device_layout,
            } = spec;

            let device_set = 16 + group_idx;
            device_blocks.push(Some(KvbmWorker::make_layout::<_, BasicMetadata>(
                device_layout,
                transfer_context.nixl_agent().as_ref(),
                device_set,
                0,
            )?));

            let host_alloc_wire = Arc::clone(&host_allocator);
            let host_layout = tier_layout_builder(&geometry, num_host_blocks)
                .build()?
                .allocate_layout(LayoutType::FullyContiguous, host_alloc_wire)
                .map_err(|e| anyhow::anyhow!("host layout for layer group {group_idx}: {e}"))?;
            if let Some(mapping) = host_allocator.io_mapping() {
                host_io_mappings.push(mapping);
            }
            let host_set = 32 + group_idx;
            host_blocks.push(Some(KvbmWorker::make_layout::<_, BasicMetadata>(
                host_layout,
                transfer_context.nixl_agent().as_ref(),
                host_set,
                0,
            )?));
        }

        if !host_io_mappings.is_empty() {
            transfer_context.set_host_tier_io_mappings(host_io_mappings.clone());
        }

        let device_data: Vec<Vec<LocalBlockData<DeviceStorage>>> = device_blocks
            .iter()
            .map(|blocks| local_data_clone(blocks.as_ref().expect("device pool")))
            .collect();
        let host_data: Vec<Vec<LocalBlockData<DevbarStorage>>> = host_blocks
            .iter()
            .map(|blocks| local_data_clone(blocks.as_ref().expect("host pool")))
            .collect();
        let disk_blocks: Vec<Option<LocalBlockList<DiskStorage>>> =
            (0..num_groups).map(|_| None).collect();

        let handler = BlockTransferHandler::new(
            device_blocks,
            host_blocks,
            disk_blocks,
            transfer_context,
            None,
            transfer::NcclConfig::disabled(),
        )?;
        Ok(TwoGroupPools {
            handler,
            device_data,
            host_data,
            io_mappings: host_io_mappings,
        })
    }

    /// The handler's own `LocalBlockData` extraction (transfer.rs's
    /// `get_local_data`), kept by the test as its observation handle over the
    /// pool-owned memory.
    fn local_data_clone<S: Storage>(
        blocks: &[Block<S, locality::Local, BasicMetadata>],
    ) -> Vec<LocalBlockData<S>> {
        blocks
            .iter()
            .map(|b| {
                let data = b.block_data() as &dyn Any;
                data.downcast_ref::<LocalBlockData<S>>()
                    .expect("local block data")
                    .clone()
            })
            .collect()
    }

    /// The stamped byte at one image position: group-distinct complements
    /// (0x3C/0xC3 first stamp, 0x5A/0xA5 second) mean a byte of one group's
    /// pattern never equals the other group's at the same index, so
    /// cross-group bleed fails the first compared byte.
    fn pattern_byte(second_stamp: bool, group: usize, block: usize, seg: usize, byte: usize) -> u8 {
        let base: u8 = if second_stamp {
            [0x5A, 0xA5][group]
        } else {
            [0x3C, 0xC3][group]
        };
        base ^ (block as u8) ^ ((seg as u8) << 5) ^ (byte as u8)
    }

    /// The expected whole-block image: the per-`(layer, outer)` segments in
    /// layer-major-then-outer order — the walk order every executor (custom
    /// kernel, staged bounce, direct-io segments) copies in, and the order a
    /// `FullyContiguous` window interleaves regions at.
    fn expected_block_image<S: Storage>(
        data: &LocalBlockData<S>,
        group: usize,
        block: usize,
        second_stamp: bool,
    ) -> Vec<u8> {
        let mut image = Vec::new();
        let mut seg = 0usize;
        let block_data = data.block_data();
        for layer_idx in 0..block_data.num_layers() {
            for outer_idx in 0..block_data.num_outer_dims() {
                let size = block_data
                    .layer_view(layer_idx, outer_idx)
                    .expect("layer view")
                    .size();
                image.extend((0..size).map(|i| pattern_byte(second_stamp, group, block, seg, i)));
                seg += 1;
            }
        }
        image
    }

    /// Stamp a pattern into one device block via per-segment H2D memcpys
    /// (device pool memory is only writable through the driver).
    fn stamp_device_block(
        data: &mut LocalBlockData<DeviceStorage>,
        group: usize,
        block: usize,
        second_stamp: bool,
    ) {
        let mut seg = 0usize;
        let block_data = data.block_data_mut();
        let num_layers = block_data.num_layers();
        let num_outer = block_data.num_outer_dims();
        for layer_idx in 0..num_layers {
            for outer_idx in 0..num_outer {
                let mut view = block_data
                    .layer_view_mut(layer_idx, outer_idx)
                    .expect("device layer view");
                let size = view.size();
                let pattern: Vec<u8> = (0..size)
                    .map(|i| pattern_byte(second_stamp, group, block, seg, i))
                    .collect();
                // SAFETY: the view covers `size` writable device bytes owned
                // by the pool layout; the pattern buffer outlives the
                // (synchronous) copy.
                unsafe {
                    assert_eq!(
                        sys::cuMemcpyHtoD_v2(
                            view.as_mut_ptr() as u64,
                            pattern.as_ptr() as *const core::ffi::c_void,
                            size
                        ),
                        sys::CUresult::CUDA_SUCCESS,
                        "stamp H2D"
                    );
                }
                seg += 1;
            }
        }
    }

    /// Read one device block's whole image back via per-segment D2H memcpys.
    fn read_device_block_image(data: &LocalBlockData<DeviceStorage>) -> Vec<u8> {
        let mut image = Vec::new();
        let block_data = data.block_data();
        for layer_idx in 0..block_data.num_layers() {
            for outer_idx in 0..block_data.num_outer_dims() {
                let view = block_data
                    .layer_view(layer_idx, outer_idx)
                    .expect("device layer view");
                let size = view.size();
                let mut buf = vec![0u8; size];
                // SAFETY: the view covers `size` readable device bytes owned
                // by the pool layout; `buf` outlives the (synchronous) copy.
                unsafe {
                    assert_eq!(
                        sys::cuMemcpyDtoH_v2(
                            buf.as_mut_ptr() as *mut core::ffi::c_void,
                            view.as_ptr() as u64,
                            size
                        ),
                        sys::CUresult::CUDA_SUCCESS,
                        "probe D2H"
                    );
                }
                image.extend_from_slice(&buf);
            }
        }
        image
    }

    /// Read one host-tier block's whole image via CPU. Every backing under
    /// test is host-accessible — mock mapping, file-backed BAR window,
    /// pinned, and the real BAR window (its CPU read-back matches the
    /// existing direct-io roundtrip test's host-VA read; the transfers
    /// themselves always go through the registered executor paths).
    fn read_host_block_image(data: &LocalBlockData<DevbarStorage>) -> Vec<u8> {
        let mut image = Vec::new();
        let block_data = data.block_data();
        for layer_idx in 0..block_data.num_layers() {
            for outer_idx in 0..block_data.num_outer_dims() {
                let view = block_data
                    .layer_view(layer_idx, outer_idx)
                    .expect("host layer view");
                let size = view.size();
                // SAFETY: the view covers `size` readable bytes of
                // host-accessible pool memory that outlives this read.
                let buf = unsafe { std::slice::from_raw_parts(view.as_ptr(), size) };
                image.extend_from_slice(buf);
            }
        }
        image
    }

    /// One group-tagged request moving `MOVED_BLOCK_IDS` between two pool
    /// tiers at identical block ids — the request shape the leader's
    /// per-group fan-out produces.
    fn group_request(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        group: usize,
    ) -> BlockTransferRequest {
        BlockTransferRequest {
            from_pool,
            to_pool,
            blocks: MOVED_BLOCK_IDS.iter().map(|&b| (b, b)).collect(),
            group,
            connector_req: None,
        }
    }

    /// Scenario half 1: stamp group-distinct patterns into each group's
    /// device blocks, offload per-group (group 0's request then group 1's —
    /// the ordering that historically froze one geometry in the process-wide
    /// dims cache), and assert the host tier received each group's bytes:
    /// per-group byte counts, byte-equality per group, no cross-group bleed.
    /// Returns false when the run soft-skipped on an unloadable fatbin —
    /// the caller must skip the onboard half too (the host tier was never
    /// filled, and the kernel would fail that half the same way).
    async fn offload_and_assert_host(
        pools: &TwoGroupPools,
        skip_if_kernel_unloadable: bool,
    ) -> bool {
        for (group, blocks) in pools.device_data.iter().enumerate() {
            for block in MOVED_BLOCK_IDS {
                // A probe clone's views address the same pool-owned memory;
                // `stamp_device_block` needs the mutable view handle.
                let mut data = blocks[block].clone();
                stamp_device_block(&mut data, group, block, false);
            }
        }
        // The stamps landed (guards the stamping helper itself).
        for (group, blocks) in pools.device_data.iter().enumerate() {
            for block in MOVED_BLOCK_IDS {
                assert_eq!(
                    read_device_block_image(&blocks[block]),
                    expected_block_image(&blocks[block], group, block, false),
                    "stamp verification, group {group} block {block}"
                );
            }
        }

        for group in 0..pools.device_data.len() {
            let request = group_request(BlockTransferPool::Device, BlockTransferPool::Host, group);
            match pools.handler.execute_transfer(request).await {
                Ok(()) => {}
                Err(e) if skip_if_kernel_unloadable && e.to_string().contains("FATBIN") => {
                    // The committed vectorized_copy fatbin is sm_100-only;
                    // on other GPUs the custom kernel cannot load. This is
                    // an environment gap, not a code failure — soft-skip
                    // (build a compatible fatbin and set DYN_FATBIN_PATH to
                    // activate the full proof on this GPU).
                    eprintln!(
                        "two_group custom-kernel run: SKIPPED (no loadable vectorized_copy \
                         fatbin for this GPU; build one and set DYN_FATBIN_PATH)"
                    );
                    return false;
                }
                Err(e) => panic!("two-group offload (group {group}) failed: {e:#}"),
            }
        }

        for (group, blocks) in pools.host_data.iter().enumerate() {
            for block in MOVED_BLOCK_IDS {
                let host = read_host_block_image(&blocks[block]);
                assert_eq!(
                    host.len(),
                    BYTES_PER_BLOCK_PER_GROUP[group],
                    "per-group byte count moved, group {group}"
                );
                assert_eq!(
                    host,
                    expected_block_image(&pools.device_data[group][block], group, block, false),
                    "host tier content after offload, group {group} block {block} \
                     (expected this group's own pattern — no cross-group bleed, no \
                     uniform-size truncation)"
                );
            }
        }

        true
    }

    /// Scenario half 2: overwrite the device blocks with a second,
    /// distinct pattern, prove the overwrite took, onboard per-group, and
    /// assert each group's ORIGINAL pattern is restored byte-exactly.
    async fn overwrite_onboard_and_assert(pools: &TwoGroupPools) {
        for (group, blocks) in pools.device_data.iter().enumerate() {
            for block in MOVED_BLOCK_IDS {
                let mut data = blocks[block].clone();
                stamp_device_block(&mut data, group, block, true);
            }
        }
        // The overwrite took — the onboard below must visibly replace it.
        for (group, blocks) in pools.device_data.iter().enumerate() {
            for block in MOVED_BLOCK_IDS {
                assert_eq!(
                    read_device_block_image(&blocks[block]),
                    expected_block_image(&blocks[block], group, block, true),
                    "overwrite verification, group {group} block {block}"
                );
            }
        }

        for group in 0..pools.device_data.len() {
            pools
                .handler
                .execute_transfer(group_request(
                    BlockTransferPool::Host,
                    BlockTransferPool::Device,
                    group,
                ))
                .await
                .unwrap_or_else(|e| panic!("two-group onboard (group {group}) failed: {e:#}"));
        }

        for (group, blocks) in pools.device_data.iter().enumerate() {
            for block in MOVED_BLOCK_IDS {
                assert_eq!(
                    read_device_block_image(&blocks[block]),
                    expected_block_image(&blocks[block], group, block, false),
                    "device tier content after onboard, group {group} block {block} \
                     (expected the group's original pattern restored exactly)"
                );
            }
        }
    }

    /// The full offload/onboard scenario, skipping the onboard half when
    /// the offload soft-skipped (an unloadable kernel fails both halves the
    /// same way, and the host tier was never filled).
    async fn run_offload_onboard_scenario(pools: &TwoGroupPools, skip_if_kernel_unloadable: bool) {
        if !offload_and_assert_host(pools, skip_if_kernel_unloadable).await {
            return;
        }
        overwrite_onboard_and_assert(pools).await;
    }

    /// The REAL construction path: `perform_allocation_and_build_handler`
    /// with a 2-group `KvbmWorkerConfig` under the mock devbar env — proves
    /// the production allocation code builds per-group pools (layouts, NIXL
    /// registration, block-set ids, sequential mock windows) and returns a
    /// working handler. Content is proven by the sibling tests, which keep
    /// observation handles this path cannot hand out (the handler is
    /// opaque).
    #[tokio::test]
    async fn two_group_real_path_handler_construction() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: devbar env vars are only read by devbar code and the
        // sibling tests sharing ENV_LOCK.
        unsafe {
            std::env::remove_var(env_devbar::DYN_KVBM_DEVBAR_VMM);
            std::env::remove_var(env_devbar::DYN_KVBM_DEVBAR_BDF);
            std::env::remove_var(env_devbar::DYN_KVBM_DEVBAR_RESOURCE_INDEX);
            std::env::remove_var(env_devbar::DYN_KVBM_DEVBAR_TRANSFERS);
            std::env::remove_var(env_devbar::DYN_KVBM_DEVBAR_MEM_TYPE);
            std::env::set_var(env_devbar::DYN_KVBM_DEVBAR_MOCK, "1");
        }

        let config = two_group_config();
        let (group_layouts, geometry) =
            build_two_group_specs(&config).expect("two-group specs derive");
        assert_eq!(geometry.bytes_per_block, 15_360);
        assert_eq!(geometry.bytes_per_block_per_group, vec![8_192, 7_168]);

        let handler = perform_allocation_and_build_handler(
            group_layouts,
            config,
            LeaderMetadata {
                num_host_blocks: NUM_HOST_BLOCKS,
                num_disk_blocks: 0,
            },
            0,
            0,
            None,
        )
        .await
        .expect("2-group handler via the real allocation path (mock env; no privileges needed)");
        drop(handler);

        unsafe { std::env::remove_var(env_devbar::DYN_KVBM_DEVBAR_MOCK) };
    }

    /// MOCK backing + the staged (`StagedCpu`) semantics: the whole
    /// offload/onboard scenario runs through the staged executor's legs —
    /// CPU window->bounce, DMA bounce->device, DMA device->bounce, CPU
    /// bounce->window — against a host-RAM window: 61ffb0ab0's staged-shape
    /// pattern elevated to the pool/executor level. The mock window is
    /// pageable host memory, so GPU-side access would fault — exactly why
    /// the mock backing only ever pairs with the staged legs here.
    #[tokio::test]
    async fn two_group_offload_onboard_roundtrip_staged_semantics() {
        let config = two_group_config();
        let pools = build_two_group_pools(
            &config,
            Arc::new(DevbarAllocator::mock(0)),
            HostTransferSemantics::StagedCpu,
            NUM_HOST_BLOCKS,
        )
        .await
        .expect("two-group pools over the mock backing");
        run_offload_onboard_scenario(&pools, false).await;
    }

    /// PINNED backing (the default host tier) + `HostPointer` semantics:
    /// both directions resolve to the CUSTOM KERNEL (device pools are
    /// LayerSeparate, never fully contiguous), so group 1's batch after
    /// group 0's exercises the per-geometry kernel-dims cache end-to-end —
    /// a frozen first-geometry cache would corrupt or fault group 1.
    /// Soft-skips when the committed sm_100-only fatbin cannot load on the
    /// local GPU (see `offload_and_assert_host`).
    #[tokio::test]
    async fn two_group_custom_kernel_roundtrip_pinned_host() {
        let config = two_group_config();
        let pools = build_two_group_pools(
            &config,
            Arc::new(DevbarAllocator::default()),
            HostTransferSemantics::HostPointer,
            NUM_HOST_BLOCKS,
        )
        .await
        .expect("two-group pools over the pinned fallback");
        run_offload_onboard_scenario(&pools, true).await;
    }

    /// File-backed Bar region + the staged semantics: real per-group
    /// sequential aperture carving (disjoint page-rounded windows from ONE
    /// `DevbarAllocator`) with zero privileges, with the backing file as
    /// the assertion surface for Task 4's allocator partitioning —
    /// attention region FIRST, mamba region SECOND, nothing past group 1's
    /// window (a uniform-size truncation or a pre-cursor aliasing bug would
    /// overwrite the wrong bytes and fail here).
    #[tokio::test]
    async fn two_group_aperture_partition_staged() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // The staged semantics and the Dram NIXL registration both read the
        // env; hold them stable for this test's construction (the real-bar
        // test writes TRANSFERS under the same lock).
        unsafe {
            std::env::remove_var(env_devbar::DYN_KVBM_DEVBAR_TRANSFERS);
            std::env::remove_var(env_devbar::DYN_KVBM_DEVBAR_MEM_TYPE);
        }

        let config = two_group_config();
        let mut aperture = tempfile::NamedTempFile::new().expect("aperture file");
        aperture
            .as_file_mut()
            .set_len(APERTURE_FILE_BYTES as u64)
            .expect("size the aperture");

        let allocator = Arc::new(DevbarAllocator::bar(aperture.path().to_path_buf(), 0, 0));
        assert_eq!(
            allocator.transfer_semantics(),
            HostTransferSemantics::StagedCpu
        );
        let pools = build_two_group_pools(
            &config,
            allocator,
            HostTransferSemantics::StagedCpu,
            NUM_HOST_BLOCKS,
        )
        .await
        .expect("two-group pools over a shared file-backed aperture");
        assert!(pools.io_mappings.is_empty(), "no registration ran");

        offload_and_assert_host(&pools, false).await;

        // Task 4's partitioning, read straight from the backing file.
        let file = std::fs::read(aperture.path()).expect("read the aperture file back");
        assert_eq!(file.len(), APERTURE_FILE_BYTES);
        let attention_window = NUM_HOST_BLOCKS * ATTN_BYTES_PER_BLOCK; // 524,288, page-multiple
        let mamba_window = NUM_HOST_BLOCKS * MAMBA_BYTES_PER_BLOCK; // 458,752, page-multiple
        let attention_offset = 0;
        let mamba_offset = attention_offset + attention_window;
        for block in MOVED_BLOCK_IDS {
            let expected = expected_block_image(&pools.device_data[0][block], 0, block, false);
            let start = attention_offset + block * ATTN_BYTES_PER_BLOCK;
            assert_eq!(
                &file[start..start + ATTN_BYTES_PER_BLOCK],
                &expected[..],
                "group 0's pattern in the ATTENTION region (carved first, offset {attention_offset})"
            );
            let expected = expected_block_image(&pools.device_data[1][block], 1, block, false);
            let start = mamba_offset + block * MAMBA_BYTES_PER_BLOCK;
            assert_eq!(
                &file[start..start + MAMBA_BYTES_PER_BLOCK],
                &expected[..],
                "group 1's pattern in the MAMBA region (carved second, offset {mamba_offset})"
            );
        }
        // Never-offloaded blocks stay zero, and nothing lands past group
        // 1's window: an oversized (uniform-size) move would bleed past the
        // mamba region into the tail.
        let untouched_zero = |span: &[u8]| span.iter().all(|&b| b == 0);
        assert!(
            untouched_zero(&file[0..ATTN_BYTES_PER_BLOCK]),
            "group 0's block 0 must be untouched"
        );
        assert!(
            untouched_zero(&file[mamba_offset..mamba_offset + MAMBA_BYTES_PER_BLOCK]),
            "group 1's block 0 must be untouched"
        );
        assert!(
            untouched_zero(&file[mamba_offset + mamba_window..]),
            "nothing may land past group 1's window"
        );

        overwrite_onboard_and_assert(&pools).await;
    }

    /// REAL BAR + `DirectIo` semantics: the IOMEMORY-registered windows'
    /// device VAs, one registration per group's carved window (two io
    /// mappings wired), with GPU-side memcpys pointer-translated per window.
    /// REQUIRES root (the registration's documented non-privileged failure
    /// is CUDA_ERROR_NOT_PERMITTED) plus the expander stand-in at
    /// 0000:41:00.0 — the same guard the existing direct-io roundtrip test
    /// uses; the node run-script skips this test by name (`--skip
    /// real_bar`) because its window runs unprivileged.
    #[tokio::test]
    async fn two_group_direct_io_offload_onboard_roundtrip_real_bar() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var(env_devbar::DYN_KVBM_DEVBAR_MEM_TYPE);
            std::env::set_var(env_devbar::DYN_KVBM_DEVBAR_TRANSFERS, "direct");
        }

        let config = two_group_config();
        let bar_path = std::path::PathBuf::from("/sys/bus/pci/devices/0000:41:00.0/resource0");
        let allocator = Arc::new(DevbarAllocator::bar(bar_path, 0, 0));
        assert_eq!(
            allocator.transfer_semantics(),
            HostTransferSemantics::DirectIo
        );
        let pools = build_two_group_pools(
            &config,
            allocator,
            HostTransferSemantics::DirectIo,
            NUM_HOST_BLOCKS_REAL_BAR,
        )
        .await
        .expect("direct alloc (root + the expander stand-in needed)");
        assert_eq!(
            pools.io_mappings.len(),
            2,
            "each group's window must carry its own io mapping"
        );

        run_offload_onboard_scenario(&pools, false).await;

        unsafe { std::env::remove_var(env_devbar::DYN_KVBM_DEVBAR_TRANSFERS) };
    }

    // ---------------------------------------------------------------
    // THE LIVE-SHAPE BASELINE (single-group): vLLM 0.29's LBNHC cache
    // [num_blocks, kv_heads, block_size, 2*head_size] (the 0.6B's real
    // [7718, 8, 16, 128]) — the geometry inference's MLA-style
    // ELSE-BRANCH (candidate = kv_heads = 8 > 2 -> outer_dim=1,
    // inner = product/page = 1024), a path no byte-parity test had
    // covered (the two-group tests all used outer=2 shapes). PINNED
    // host storage + HostPointer semantics — the baseline D2H/H2D path
    // with no devbar involvement. Stamp -> offload -> verify the host
    // tier -> overwrite -> onboard -> byte-compare the restore.
    // ---------------------------------------------------------------
    #[tokio::test]
    async fn lbnhc_single_group_pinned_offload_onboard_byte_parity() {
        let num_blocks = 64;
        let kv_heads = 8;
        let tokens = 16; // page_size
        let kv_and_head = 128; // 2 * head_size (K|V concatenated)

        let tensors: Vec<Arc<dyn TorchTensor>> = (0..4)
            .map(|_| -> Arc<dyn TorchTensor> {
                Arc::new(FakeLayerTensor::new(
                    vec![num_blocks, kv_heads, tokens, kv_and_head],
                    2, // bf16
                ))
            })
            .collect();

        // The LEGACY registration's inference for this shape derives
        // (verified by a CPU unit test): candidate = shape[1] = 8 > 2 ->
        // the MLA-style else-branch: outer_dim=1, inner_dim =
        // (8*16*128)/16 = 1024. The pool test provides that derived
        // geometry EXPLICITLY as one group (build_two_group_specs only
        // covers the grouped arm; the legacy path synthesizes the same
        // single group in KvbmWorker::new).
        let mut group = LayerGroup::attention(4, tokens, 1, 1024, 2);
        group.tensor_indices = vec![0, 1, 2, 3];
        let config = KvbmWorkerConfig::builder()
            .cancel_token(CancellationToken::new())
            .num_device_blocks(num_blocks)
            .page_size(tokens)
            .tensors(tensors)
            .device_id(0)
            .dtype_width_bytes(2)
            .groups(vec![group])
            .build()
            .expect("LBNHC single-group config builds");

        // The pinned host fallback — the baseline tier-2 (no devbar env).
        let pools = build_two_group_pools(
            &config,
            Arc::new(DevbarAllocator::default()),
            HostTransferSemantics::HostPointer,
            num_blocks,
        )
        .await
        .expect("LBNHC single-group pools over pinned host");
        assert_eq!(pools.device_data.len(), 1, "single group");
        assert_eq!(pools.host_data.len(), 1);

        // Stamp DISTINCT patterns per (block, layer) and verify the stamp.
        for block in MOVED_BLOCK_IDS {
            let mut data = pools.device_data[0][block].clone();
            stamp_device_block(&mut data, 0, block, false);
        }
        for block in MOVED_BLOCK_IDS {
            assert_eq!(
                read_device_block_image(&pools.device_data[0][block]),
                expected_block_image(&pools.device_data[0][block], 0, block, false),
                "stamp verification, block {block}"
            );
        }

        // D2H: offload and verify the PINNED host tier received the exact
        // bytes — layer-major-then-outer (outer=1: one 32,768 B segment
        // per layer), the exact image the device blocks hold.
        pools
            .handler
            .execute_transfer(group_request(
                BlockTransferPool::Device,
                BlockTransferPool::Host,
                0,
            ))
            .await
            .expect("LBNHC single-group offload");

        for block in MOVED_BLOCK_IDS {
            let host = read_host_block_image(&pools.host_data[0][block]);
            assert_eq!(
                host.len(),
                4 * kv_heads * tokens * kv_and_head * 2,
                "per-block byte count (4 layers x 32,768 B)"
            );
            assert_eq!(
                host,
                expected_block_image(&pools.device_data[0][block], 0, block, false),
                "PINNED host tier content after offload, block {block} — the \
                 offload must store the block's exact bytes"
            );
        }

        // H2D: overwrite the device blocks, onboard, byte-compare the
        // restore against the ORIGINAL pattern.
        for block in MOVED_BLOCK_IDS {
            let mut data = pools.device_data[0][block].clone();
            stamp_device_block(&mut data, 0, block, true);
        }
        pools
            .handler
            .execute_transfer(group_request(
                BlockTransferPool::Host,
                BlockTransferPool::Device,
                0,
            ))
            .await
            .expect("LBNHC single-group onboard");

        for block in MOVED_BLOCK_IDS {
            assert_eq!(
                read_device_block_image(&pools.device_data[0][block]),
                expected_block_image(&pools.device_data[0][block], 0, block, false),
                "device content after onboard, block {block} — the restore \
                 must return the block's original bytes (the determinism \
                 probe's requirement)"
            );
        }
    }
}

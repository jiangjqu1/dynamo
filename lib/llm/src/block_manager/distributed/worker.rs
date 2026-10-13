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
    storage::{DevbarAllocator, DeviceAllocator, DeviceStorage, DiskAllocator, torch::TorchTensor},
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
    device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
    mut layout_builder: LayoutConfigBuilder,
    worker_config: KvbmWorkerConfig,
    leader_meta: LeaderMetadata,
    worker_id: usize,
    device_id: usize,
    scheduler_client: Option<TransferSchedulerClient>,
) -> anyhow::Result<BlockTransferHandler> {
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
        num_outer_components: device_layout.config().outer_dim,
        num_layers: device_layout.config().num_layers,
    };
    // Host (G2 / tier-2) allocator, built before the transfer context so its
    // backing's transfer_semantics() can set the context's host-transfer
    // semantics: DeviceAlias for a VMM region, StagedCpu for a Bar/Bdf region
    // on a raw sysfs MMIO aperture (staged default; `direct` opt-in),
    // HostPointer otherwise. The pool itself is only allocated below when
    // should_allocate_offload.
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

    // device - always allocated on all ranks
    let device_blocks = Some(KvbmWorker::make_layout::<_, BasicMetadata>(
        device_layout,
        transfer_context.nixl_agent().as_ref(),
        0,
        worker_id,
    )?);

    // host (G2 / tier-2) - only allocated if should_allocate_offload.
    // Backed by a BAR-mapped devbar region when DYN_KVBM_DEVBAR_* is configured,
    // otherwise standard pinned host memory (default deployments unchanged).
    let host_blocks = if should_allocate_offload && leader_meta.num_host_blocks > 0 {
        // The layout takes the Arc by value; clone so the io-mapping stash
        // (filled by the allocate inside) stays readable after the call.
        let host_alloc_wire = Arc::clone(&host_allocator);
        let host_layout = layout_builder
            .num_blocks(leader_meta.num_host_blocks)
            .build()?
            .allocate_layout(worker_config.host_layout_type, host_alloc_wire)?;
        // Direct transfers: wire the BAR window's io mapping (host base,
        // device base, length) into the context for the executor's pointer
        // translation. Present only when the IOMEMORY registration succeeded.
        if let Some(mapping) = host_allocator.io_mapping() {
            transfer_context.set_host_tier_io_mapping(mapping);
        }
        Some(KvbmWorker::make_layout::<_, BasicMetadata>(
            host_layout,
            transfer_context.nixl_agent().as_ref(),
            1,
            worker_id,
        )?)
    } else {
        None
    };
    // disk (G3) - only allocated if should_allocate_offload
    let disk_blocks = if should_allocate_offload && leader_meta.num_disk_blocks > 0 {
        let disk_allocator = Arc::new(DiskAllocator::from_env()?);
        let disk_layout = layout_builder
            .num_blocks(leader_meta.num_disk_blocks)
            .build()?
            .allocate_layout(worker_config.disk_layout_type, disk_allocator)?;
        Some(KvbmWorker::make_layout::<_, BasicMetadata>(
            disk_layout,
            transfer_context.nixl_agent().as_ref(),
            2,
            worker_id,
        )?)
    } else {
        None
    };

    // Single-group legacy construction: one pool list per tier. The real
    // per-group construction lands with the grouped pool path below.
    let handler = BlockTransferHandler::new(
        vec![device_blocks],
        vec![host_blocks],
        vec![disk_blocks],
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
    device_layout: Mutex<Option<Box<dyn NixlLayout<StorageType = DeviceStorage>>>>,
    layout_builder: LayoutConfigBuilder,
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

        // Take device_layout once.
        let dev_layout = {
            let mut guard = self.device_layout.lock().await;
            match guard.take() {
                Some(d) => d,
                None => {
                    tracing::warn!("leader_metadata: device_layout already consumed; dropping");
                    return Ok(());
                }
            }
        };

        // Capture what we need and run allocation in the background.
        let layout_builder = self.layout_builder.clone();
        let worker_config = self.worker_config.clone();
        let worker_id = self.worker_id;
        let device_id = self.device_id;
        let scheduler_client = self.scheduler_client.clone();
        let handler_cell = self.handler_cell.clone();
        let handler_tx = self.handler_tx.clone();
        let state = self.state.clone();

        tokio::spawn(async move {
            match perform_allocation_and_build_handler(
                dev_layout,
                layout_builder,
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
    /// Indices into the registration-order tensor list; strictly increasing,
    /// and across all groups a partition of `0..tensors.len()`.
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
        let (outer_dim, inner_dim) =
            infer_layer_separate_dims(&shape, group.page_bytes, true, explicit);
        GroupGeometry {
            layout_type: group.layout_type(),
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

        // Grouped configs: the groups must partition the registered tensor
        // list in registration order, and `active_save_layers` is bounded by
        // the number of registered tensors. Ungrouped (legacy) configs are
        // not subject to the bound.
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

            // A single ordered pass: the concatenation of the groups'
            // indices must be exactly 0..tensors.len(), which enforces
            // strict increase, disjointness, and registration order.
            let mut expected = 0usize;
            for (group_idx, group) in self.groups.iter().enumerate() {
                for &tensor_idx in &group.tensor_indices {
                    if tensor_idx != expected {
                        anyhow::bail!(
                            "LayerGroup {} tensor indices must be strictly increasing and follow registration order; expected index {}, got {}",
                            group_idx,
                            expected,
                            tensor_idx
                        );
                    }
                    expected += 1;
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

        if shape.len() < 3 {
            return Err(anyhow::anyhow!(format!(
                "Unsupported kv cache layout. Got shape: {:?}",
                shape
            )));
        }

        let (
            layout_type,
            num_layers,
            outer_dim,
            inner_dim,
            page_size,
            dtype_width_bytes,
            bytes_per_block,
            bytes_per_block_per_group,
        ) = if config.groups.is_empty() {
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

            (
                layout_type,
                num_layers,
                outer_dim,
                inner_dim,
                config.page_size,
                config.dtype_width_bytes,
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

            // STAGING (Task 4): pool construction still consumes only the
            // FIRST group's geometry; per-group layouts are not threaded
            // through the transfer handler yet, and create_layout currently
            // consumes the first group's layout flavor with ALL tensors.
            // Both are cleared in Task 4.
            let first = &geometries[0];
            (
                first.layout_type,
                first.num_layers,
                first.outer_dim,
                first.inner_dim,
                first.page_size,
                first.dtype_width_bytes,
                bytes_per_block,
                bytes_per_block_per_group,
            )
        };

        let mut layout_builder_instance = LayoutConfigBuilder::default();
        let layout_builder = layout_builder_instance
            .num_layers(num_layers)
            .outer_dim(outer_dim)
            .page_size(page_size)
            .inner_dim(inner_dim)
            .dtype_width_bytes(dtype_width_bytes);

        let device_layout = layout_builder
            .num_blocks(config.num_device_blocks)
            .build()?
            .create_layout(layout_type, device_tensors)?;

        let layout_builder = layout_builder.clone();

        let (task, handler_rx) = if layout_blocking {
            Self::run_blocking_layout_initialization(
                config,
                bytes_per_block,
                bytes_per_block_per_group,
                device_layout,
                layout_builder,
                layout_type,
            )
            .await?
        } else {
            Self::run_non_blocking_layout_initialization(
                config,
                bytes_per_block,
                bytes_per_block_per_group,
                device_layout,
                layout_builder,
                layout_type,
            )
            .await?
        };

        Ok(Self {
            task: Some(task),
            block_transfer_handler_rx: Some(handler_rx),
        })
    }

    async fn run_blocking_layout_initialization(
        config: KvbmWorkerConfig,
        bytes_per_block: usize,
        bytes_per_block_per_group: Vec<usize>,
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        layout_builder: LayoutConfigBuilder,
        layout_type: LayoutType,
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

        let worker_config = config.clone();
        // start background worker task to do layout allocation for host or disk
        let task = CriticalTaskExecutionHandle::new(
            move |cancel_token| {
                KvbmWorker::worker_task(
                    device_layout,
                    layout_builder,
                    layout_type,
                    worker_config,
                    cancel_token,
                    handler_tx_cell,
                    layout_ready_tx_cell,
                    scheduler_client,
                    bytes_per_block,
                    bytes_per_block_per_group,
                )
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
        bytes_per_block: usize,
        bytes_per_block_per_group: Vec<usize>,
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage> + Send + 'static>,
        layout_builder: LayoutConfigBuilder,
        layout_type: LayoutType,
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
                let cfg = worker_config.clone();
                let scheduler = scheduler_client.clone();

                async move {
                    // Start the long-running worker.
                    let dev_layout = device_layout; // moved in
                    let lb = layout_builder; // moved in
                    let lt = layout_type; // moved in

                    let worker_fut = KvbmWorker::worker_task(
                        dev_layout,
                        lb,
                        lt,
                        cfg.clone(),
                        ct.clone(),
                        handler_tx_cell,
                        layout_ready_tx_cell,
                        scheduler,
                        bytes_per_block,
                        bytes_per_block_per_group,
                    );

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

    #[allow(clippy::too_many_arguments)]
    async fn worker_task(
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        layout_builder: LayoutConfigBuilder,
        _device_layout_type: LayoutType,
        config: KvbmWorkerConfig,
        cancel_token: CancellationToken,
        handler_tx: Arc<Mutex<Option<oneshot::Sender<BlockTransferHandler>>>>,
        layout_ready_tx: tokio::sync::Mutex<Option<oneshot::Sender<String>>>,
        scheduler_client: Option<TransferSchedulerClient>,
        bytes_per_block: usize,
        bytes_per_block_per_group: Vec<usize>,
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
                bytes_per_block,
                bytes_per_block_per_group,
            }) as Arc<dyn Handler>,
        );

        handlers.insert(
            ZMQ_LEADER_METADATA_MESSAGE.to_string(),
            Arc::new(LeaderMetadataHandler {
                state: state.clone(),
                device_layout: tokio::sync::Mutex::new(Some(device_layout)), // moved in
                layout_builder,                                              // moved
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
        // attention group: 4 layers, K/V outer=2, page 784 tokens, inner 512, fp8(1B)
        let attn = LayerGroup::attention(4, 784, 2, 512, 1);
        assert_eq!(attn.bytes_per_block(), 4 * 2 * 784 * 512 * 1);
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
        let groups = vec![
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
    fn validate_groups_reject_non_registration_order() {
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
            err.contains("registration order"),
            "expected registration-order error, got: {err}"
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
        let geometry = derive_group_geometry(&group, &tensors, (None, None));
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
        let geometry = derive_group_geometry(&group, &tensors, (None, None));
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

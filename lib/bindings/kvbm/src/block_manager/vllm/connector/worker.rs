// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::block_manager::connector::protocol::{RequestType, TransferType};
use dynamo_llm::block_manager::connector::scheduler::{
    Scheduler, TransferSchedulerClient, WorkerSchedulerClient,
};

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use super::*;
use crate::block_manager::distributed::{get_leader_zmq_ack_url, get_leader_zmq_pub_url};
use crate::{block_manager::distributed::VllmTensor, to_pyerr};

use crate::block_manager::distributed::PyLayoutType;
use crate::{
    extract_distributed_runtime_from_obj, get_current_cancel_token, get_current_tokio_handle,
};
use anyhow;
use dynamo_llm::block_manager::distributed::{KvbmWorker, KvbmWorkerConfig, LayerGroup};
use dynamo_llm::block_manager::layout::LayoutType;
use dynamo_llm::block_manager::storage::torch::TorchTensor;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

pub trait Worker: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: Vec<(String, Arc<VllmTensor>)>,
        raw_event_handles: Vec<u64>,
        device_layout_type: Option<LayoutType>,
        host_layout_type: Option<LayoutType>,
        disk_layout_type: Option<LayoutType>,
        outer_dim: Option<usize>,
        inner_dim: Option<usize>,
        group_of_layer: Option<Vec<usize>>,
        group_specs: Option<Vec<(bool, Option<usize>)>>,
        active_save_layers: Option<usize>,
    ) -> anyhow::Result<()>;

    fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> anyhow::Result<()>;

    fn clear_connector_metadata(&mut self);

    fn save_kv_layer(&mut self, layer_name: String) -> anyhow::Result<()>;

    /// Hybrid (HMA) save trigger: vLLM calls this once per step after the
    /// forward pass, with a freshly recorded event whose completion covers
    /// ALL registered layers (mamba/GDN state bytes are only final then).
    /// Attention-only registrations ignore it — their offload enqueue
    /// already happened at the last `save_kv_layer`.
    fn wait_for_save(&mut self, event_handle: u64) -> anyhow::Result<()>;

    fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>);
}

pub struct KvConnectorWorker {
    _drt: Option<Arc<DistributedRuntime>>,
    kvbm_worker: OnceLock<KvbmWorker>,
    connector: WorkerSchedulerClient,
    transfer_client: TransferSchedulerClient,

    kv_cache_layers: Vec<(String, Arc<VllmTensor>)>,

    /// Map of request id to inflight load requests
    maybe_finished_onboarding: HashSet<String>,

    /// Map of request id to inflight finished requests
    maybe_finished_offloading: HashSet<String>,

    /// For now, offloading operations will be enqueued at the end of the forward pass
    offloading_operations: Vec<WorkerTransferRequest>,

    bound: bool,
    iteration: u64,
    layers_complete: usize,

    /// Hybrid (HMA) trigger: the number of layers that call
    /// `save_kv_layer` (attention layers; mamba/GDN layers never do).
    /// `None` = attention-only legacy behavior: trigger and enqueue at
    /// the last registered layer. `Some(n)` = sync the last attention
    /// layer's event when the n-th save arrives, but defer the offload
    /// enqueue to `wait_for_save` (post-forward, when mamba bytes are
    /// final).
    active_save_layers: Option<usize>,

    /// cuda events created by the python side
    layer_events: Vec<u64>,
}

impl KvConnectorWorker {
    fn new(drt: Option<Arc<DistributedRuntime>>, vllm_worker_id: String) -> anyhow::Result<Self> {
        let runtime = get_current_tokio_handle();

        let (scheduler, worker_client, transfer_client) =
            Scheduler::new(get_current_cancel_token());

        CriticalTaskExecutionHandle::new_with_runtime(
            move |_| {
                let mut scheduler = scheduler;
                async move { scheduler.run().await }
            },
            get_current_cancel_token(),
            "kv-connector-scheduler-task",
            &runtime,
        )?
        .detach();

        tracing::info!(
            "KvConnectorWorker initialized with worker_id: {}",
            vllm_worker_id
        );

        Ok(Self {
            _drt: drt,
            kvbm_worker: OnceLock::new(),
            connector: worker_client,
            transfer_client,
            maybe_finished_onboarding: HashSet::new(),
            maybe_finished_offloading: HashSet::new(),
            offloading_operations: Vec::new(),
            bound: false,
            iteration: 0,
            layers_complete: 0,
            active_save_layers: None,
            kv_cache_layers: Vec::new(),
            layer_events: Vec::new(),
        })
    }

    /// Hand the iteration's offload operations to the scheduler. The
    /// operations are taken (not borrowed): once enqueued, the field must
    /// be empty so a missing trigger never double-enqueues.
    fn enqueue_offload_operations(&mut self) {
        let offloading_operations = std::mem::take(&mut self.offloading_operations);

        tracing::trace!(
            iteration = self.iteration,
            num_operations = offloading_operations.len(),
            "Enqueuing {} offload operations",
            offloading_operations.len()
        );

        for operation in &offloading_operations {
            tracing::debug!(
                request_id = %operation.request_id,
                operation_id = %operation.uuid,
                "Enqueuing offload operation to scheduler"
            );
            self.connector.enqueue_request(operation.clone());
        }
    }

    /// Enqueue offload operations still held from a previous iteration.
    ///
    /// Hybrid registrations defer the enqueue to `wait_for_save`; a step
    /// that never runs a forward pass (e.g. `no_forward`) skips it, so
    /// the held operations would otherwise be silently dropped when the
    /// next iteration's metadata overwrites them. No fresh event sync is
    /// possible at bind time — the enqueue proceeds unsynced and the
    /// affected requests are named in a warning so the staleness is
    /// visible.
    ///
    /// A slot whose held stores were never enqueued is vacuously
    /// complete (un-enqueued holds are not in the slot's operation
    /// list), so `get_finished` may already have reaped it during the
    /// forward-less step. Enqueuing into a reaped slot would panic the
    /// scheduler client's hard `expect`, so those operations are dropped
    /// with a warning naming the request and operation instead.
    fn flush_held_offload_operations(&mut self) {
        if self.offloading_operations.is_empty() {
            return;
        }
        let held = std::mem::take(&mut self.offloading_operations);
        let request_ids: Vec<&str> = held.iter().map(|op| op.request_id.as_str()).collect();
        tracing::warn!(
            iteration = self.iteration,
            request_ids = ?request_ids,
            num_operations = request_ids.len(),
            "enqueuing offload operations held from a previous iteration without a fresh event sync (that iteration never reached wait_for_save)"
        );
        for operation in held {
            if !self.connector.has_slot(&operation.request_id) {
                tracing::warn!(
                    request_id = %operation.request_id,
                    operation_id = %operation.uuid,
                    "slot already reaped; dropping stale held offload op"
                );
                continue;
            }
            tracing::debug!(
                request_id = %operation.request_id,
                operation_id = %operation.uuid,
                "Enqueuing held offload operation to scheduler"
            );
            self.connector.enqueue_request(operation);
        }
    }
}

/// Build the HMA [`LayerGroup`]s for a hybrid registration.
///
/// `group_of_layer` (parallel to the registered tensors, in registration
/// order) assigns each tensor to a group id; `group_specs` carries
/// `(is_mamba, page_bytes)` per group in group-id order. `None` or empty
/// inputs produce no groups, keeping the legacy single-geometry path.
///
/// Mamba groups carry their full geometry: `page_bytes` is the physical
/// per-block row size `C` of the dense int8 `[B, C]` state tensor, which
/// Python derives from the tensor itself (vLLM's spec page_size is padded
/// metadata, not the physical row). Attention groups only need the
/// page-token count — their outer/inner dims are inferred per group from
/// the group's first registered tensor (or taken from the explicit
/// config-level dims); the outer/inner values stored here are inert
/// placeholders that exist only so `validate()`'s non-zero
/// `bytes_per_block` rule passes.
#[allow(clippy::too_many_arguments)]
fn build_layer_groups(
    group_of_layer: Option<&[usize]>,
    group_specs: Option<&[(bool, Option<usize>)]>,
    num_registered: usize,
    page_size: usize,
    dtype_width_bytes: usize,
    outer_dim: Option<usize>,
    inner_dim: Option<usize>,
) -> anyhow::Result<Vec<LayerGroup>> {
    let (Some(group_of_layer), Some(group_specs)) = (group_of_layer, group_specs) else {
        return Ok(Vec::new());
    };
    if group_of_layer.is_empty() || group_specs.is_empty() {
        return Ok(Vec::new());
    }
    if group_of_layer.len() != num_registered {
        anyhow::bail!(
            "group_of_layer has {} entries but {} tensors were registered",
            group_of_layer.len(),
            num_registered
        );
    }

    // Collect each group's tensor indices, preserving registration order.
    // Groups may interleave (e.g. attention layers inside a mamba span).
    let mut indices_per_group = vec![Vec::<usize>::new(); group_specs.len()];
    for (tensor_idx, &group_id) in group_of_layer.iter().enumerate() {
        let Some(indices) = indices_per_group.get_mut(group_id) else {
            anyhow::bail!(
                "group_of_layer references group id {} but only {} groups were provided",
                group_id,
                group_specs.len()
            );
        };
        indices.push(tensor_idx);
    }

    let mut groups = Vec::with_capacity(group_specs.len());
    for (group_id, (is_mamba, page_bytes)) in group_specs.iter().enumerate() {
        let tensor_indices = std::mem::take(&mut indices_per_group[group_id]);
        if *is_mamba {
            if tensor_indices.is_empty() && page_bytes.is_none() {
                // An empty group has no state tensor to derive the
                // physical row size from; name the emptiness instead of
                // letting the page_bytes requirement misreport the
                // problem. (Non-empty builds pass empty groups through —
                // KvbmWorkerConfig::validate() owns their rejection with
                // a "has no tensor indices" error that names the group.)
                anyhow::bail!(
                    "mamba layer group {group_id} has no layers in group_of_layer; an empty mamba group has no state tensor to derive page_bytes from"
                );
            }
            let Some(page_bytes) = page_bytes else {
                anyhow::bail!(
                    "mamba layer group {group_id} requires page_bytes (the physical per-block row size of its state tensor)"
                );
            };
            let mut group = LayerGroup::mamba(tensor_indices.len(), *page_bytes);
            group.tensor_indices = tensor_indices;
            groups.push(group);
        } else {
            let page_tokens = page_bytes.unwrap_or(page_size);
            let mut group = LayerGroup::attention(
                tensor_indices.len(),
                page_tokens,
                outer_dim.unwrap_or(1),
                inner_dim.unwrap_or(1),
                dtype_width_bytes,
            );
            group.tensor_indices = tensor_indices;
            groups.push(group);
        }
    }
    Ok(groups)
}

impl Worker for KvConnectorWorker {
    /// Registers the KV caches with the KVBM worker.
    ///
    /// The Dynamo KVBM worker is lazily initialized when the first KV cache is registered.
    /// This process establishes a connection between all KVBM workers and the leader.
    #[allow(clippy::too_many_arguments)]
    fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: Vec<(String, Arc<VllmTensor>)>,
        raw_event_handles: Vec<u64>,
        device_layout_type: Option<LayoutType>,
        host_layout_type: Option<LayoutType>,
        disk_layout_type: Option<LayoutType>,
        outer_dim: Option<usize>,
        inner_dim: Option<usize>,
        group_of_layer: Option<Vec<usize>>,
        group_specs: Option<Vec<(bool, Option<usize>)>>,
        active_save_layers: Option<usize>,
    ) -> anyhow::Result<()> {
        if self.kvbm_worker.get().is_some() {
            tracing::warn!("kvbm worker already registered");
            return Err(anyhow::anyhow!("kvbm worker already registered"));
        }

        assert_eq!(
            kv_caches.len(),
            raw_event_handles.len(),
            "kv_caches and raw_event_handles must have the same length"
        );

        // Process kv_caches in layer execution order (already sorted by layer index)
        let mut vllm_tensors = Vec::new();
        let mut first_tensor_shape: Option<Vec<usize>> = None;

        for (layer_name, vllm_tensor) in kv_caches {
            tracing::trace!("Registering KV cache layer: {layer_name}, tensor: {vllm_tensor:?}");

            // Capture the shape of the first tensor for layout detection
            if first_tensor_shape.is_none() {
                first_tensor_shape = Some(vllm_tensor.shape());
            }

            // Store for later lookup by name
            self.kv_cache_layers.push((layer_name, vllm_tensor.clone()));

            // Build ordered tensor list for worker config
            vllm_tensors.push(vllm_tensor as Arc<dyn TorchTensor>);
        }

        self.layer_events = raw_event_handles;

        // Hybrid (HMA) handoff: build the per-group layer geometry so the
        // worker derives per-group layouts/pools, and remember the
        // attention-layer count for the save trigger. None/empty inputs
        // keep the legacy single-group path byte-identical.
        let groups = build_layer_groups(
            group_of_layer.as_deref(),
            group_specs.as_deref(),
            vllm_tensors.len(),
            page_size,
            dtype_width_bytes,
            outer_dim,
            inner_dim,
        )?;
        self.active_save_layers = active_save_layers;
        if let Some(active_save_layers) = active_save_layers {
            tracing::info!(
                active_save_layers,
                num_groups = groups.len(),
                "hybrid registration: save trigger moved to wait_for_save"
            );
        }

        // Auto-detect device layout type if not explicitly provided
        let detected_device_layout_type = match device_layout_type {
            Some(layout) => layout,
            None => {
                if let Some(ref shape) = first_tensor_shape {
                    match LayoutType::layer_separate_auto(shape, num_device_blocks) {
                        Ok(detected) => {
                            tracing::info!(
                                "Auto-detected device layout from tensor shape: {:?}",
                                detected
                            );
                            detected
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to auto-detect layout from shape {:?}: {}. Using default.",
                                shape,
                                e
                            );
                            LayoutType::layer_separate_auto_default()
                        }
                    }
                } else {
                    tracing::warn!("No tensors available for layout detection. Using default.");
                    LayoutType::layer_separate_auto_default()
                }
            }
        };

        let mut config_builder = KvbmWorkerConfig::builder()
            .cancel_token(get_current_cancel_token())
            .num_device_blocks(num_device_blocks)
            .page_size(page_size)
            .tensors(vllm_tensors)
            .device_id(device_id)
            .dtype_width_bytes(dtype_width_bytes)
            .scheduler_client(Some(self.transfer_client.clone()))
            .device_layout_type(detected_device_layout_type)
            .host_layout_type(host_layout_type.unwrap_or(LayoutType::FullyContiguous))
            .disk_layout_type(disk_layout_type.unwrap_or(LayoutType::FullyContiguous))
            .leader_pub_url(get_leader_zmq_pub_url())
            .leader_ack_url(get_leader_zmq_ack_url());
        if let Some(od) = outer_dim {
            config_builder = config_builder.outer_dim(Some(od));
        }
        if let Some(id) = inner_dim {
            config_builder = config_builder.inner_dim(Some(id));
        }
        let config = config_builder
            .groups(groups)
            .active_save_layers(active_save_layers)
            .build()?;

        let worker = get_current_tokio_handle().block_on(async move {
            let worker = KvbmWorker::new(config, false).await?;
            anyhow::Ok(worker)
        })?;

        self.kvbm_worker
            .set(worker)
            .map_err(|_| anyhow::anyhow!("failed to set kvbm worker"))?;

        Ok(())
    }

    /// Loads the metadata from the leader.
    /// This action translates the metadata into a set of actions that the worker will perform.
    /// All actions much be assigned to a slot before [`KvConnectorWorker::clear_metadata`] is called.
    fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> anyhow::Result<()> {
        // debug_assert!(!self.bound, "connector metadata already bound");
        let metadata: ConnectorMetadata = serde_json::from_slice(&metadata)?;
        self.bound = true;
        self.iteration = metadata.iteration;
        self.layers_complete = 0;
        tracing::debug!(
            iteration = self.iteration,
            "bound new metadata: {metadata:#?}"
        );

        self.connector.start_next_iteration()?;

        debug_assert_eq!(
            self.connector.iteration(),
            metadata.iteration,
            "iteration mismatch"
        );

        // self.engine_tx
        //     .send(EngineMessage::UpdateIteration(self.iteration))
        //     .map_err(to_pyerr)?;

        // local actions
        // - create a request slot for each new request
        // - for each action in the metadata, add the action to the request slot
        // - send the list of actions to the engine to track completion

        for slot_info in &metadata.new_slots {
            debug_assert!(
                !self.connector.has_slot(&slot_info.request_id),
                "slot already exists"
            );
            // Create slot with expected immediate ops count BEFORE any operations arrive.
            // This ensures proper completion tracking and avoids race conditions in TP>1.
            self.connector.create_slot_with_immediate_ops(
                slot_info.request_id.clone(),
                slot_info.expected_immediate_ops,
            )?;
        }

        let mut onboarding_operations = Vec::new();
        let mut offloading_operations = Vec::new();
        let mut flush_operations = Vec::new();

        for operation in metadata.operations {
            tracing::debug!(
                request_id = operation.request_id, operation_id = %operation.uuid,
                "adding operation to slot: {operation:#?}"
            );

            match operation.transfer_type {
                TransferType::Load => onboarding_operations.push(operation),
                TransferType::Store => match operation.request_type {
                    // Finish-flush Stores (hybrid multi-group): the data is
                    // final at request finish and no forward pass remains to
                    // gate them — enqueue at bind (like Immediate Loads).
                    // A Scheduled hold would never release on the idle
                    // step (no wait_for_save), the slot would be reaped as
                    // vacuously complete, and the parked ZMQ transfer would
                    // deadlock the transfer engine.
                    RequestType::Flush => flush_operations.push(operation),
                    _ => offloading_operations.push(operation),
                },
            }
        }

        // immediately enqueue the onboarding operations
        for operation in onboarding_operations {
            let request_id = operation.request_id.clone();
            self.connector.enqueue_request(operation);
            self.maybe_finished_onboarding.insert(request_id);
        }

        // immediately enqueue the finish-flush operations (see above)
        for operation in flush_operations {
            let request_id = operation.request_id.clone();
            self.connector.enqueue_request(operation);
            self.maybe_finished_offloading.insert(request_id);
        }

        // Hybrid only: a prior iteration may still HOLD offload operations
        // (a step without a forward pass never reaches wait_for_save).
        // Enqueue them before the new metadata overwrites the field, or
        // their slots would never complete. Legacy registrations cannot
        // hold operations — their enqueue already happened at the last
        // save_kv_layer — so the legacy path stays byte-identical.
        if self.active_save_layers.is_some() {
            self.flush_held_offload_operations();
        }

        self.offloading_operations = offloading_operations;

        Ok(())
    }

    /// Clears the connector metadata and marks the iteration as complete.
    fn clear_connector_metadata(&mut self) {
        tracing::debug!(iteration = self.iteration, "clearing connector metadata");
        debug_assert!(self.bound, "connector metadata not bound");
        self.bound = false;
        self.iteration = 0; // always reset; leader drives the counter
        self.layers_complete = 0;
        self.connector
            .mark_iteration_complete()
            .expect("failed to mark iteration complete");
    }

    /// Trigger layer-wise completion signals.
    /// Trigger block-wise completion signals after the last layer.
    ///
    /// Attention-only (legacy): when the last registered layer's save
    /// arrives, sync its event and enqueue the offload operations — the
    /// exact historical path. Hybrid (HMA): mamba/GDN layers never call
    /// this, so the last ATTENTION layer's save arrives before mamba
    /// bytes are final. Fire the trigger (log that the offload
    /// operations are held) but do NOT sync any event here —
    /// `layers_complete - 1` is a registration-position index, which for
    /// interleaved layouts points at a mamba-layer event, and nothing
    /// consumes that fence: the enqueue goes to
    /// [`Worker::wait_for_save`], whose fresh post-forward event is the
    /// real ordering guarantee.
    fn save_kv_layer(&mut self, _layer_name: String) -> anyhow::Result<()> {
        self.layers_complete += 1;
        let trigger_at = self
            .active_save_layers
            .unwrap_or(self.kv_cache_layers.len());
        tracing::debug!(
            iteration = self.iteration,
            layers_complete = self.layers_complete,
            trigger_at,
            total_layers = self.kv_cache_layers.len(),
            pending_offload_ops = self.offloading_operations.len(),
            "save_kv_layer called"
        );
        if self.layers_complete == trigger_at {
            if self.active_save_layers.is_none() {
                // Attention-only (legacy): sync the last layer's event
                // and enqueue — byte-identical to the historical path.
                event_sync_blocking(self.layer_events[self.layers_complete - 1]);
                self.enqueue_offload_operations();
            } else {
                // Hybrid: hold — no event sync at the trigger (the
                // index above is a registration position, not an
                // attention-layer position); wait_for_save enqueues
                // after the full forward pass behind a fresh event
                // that covers every registered layer.
                tracing::debug!(
                    iteration = self.iteration,
                    trigger_at,
                    pending_offload_ops = self.offloading_operations.len(),
                    "hybrid save trigger fired; offload ops held for wait_for_save"
                );
            }
        }
        Ok(())
    }

    /// Hybrid save trigger (see [`Worker::wait_for_save`]). The event was
    /// freshly recorded by Python after the forward pass, so its
    /// completion covers every registered layer, mamba included; sync it
    /// and only then enqueue the held offload operations. Attention-only
    /// registrations never reach the enqueue here (no-op).
    fn wait_for_save(&mut self, event_handle: u64) -> anyhow::Result<()> {
        if self.active_save_layers.is_some() {
            event_sync_blocking(event_handle);
            self.enqueue_offload_operations();
        }
        Ok(())
    }

    fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>) {
        tracing::debug!(
            iteration = self.iteration,
            "Getting finished requests: {finished_requests:?}"
        );

        // we do not have to visit every slot on every pass, just slots we are waiting on
        //
        // there are two conditions where we would be waiting:
        // 1. if we have requested a load, we need to wait for it to complete
        //    - the load request would come in via the metadata this is processsed in the bind
        // 2. if we have requested a finished event, then we need to await for all outstanding
        //    operations to complete -- either by finishing or being cancelled
        //    - the finish request is triggered by this function, it is not seen in the metadata
        //
        // under each scenario, we mark the `maybe_loading_finished` and `maybe_finished_offloading` hashsets with
        // the request id
        //
        // on each forward pass we visit the maybe slots to see if they are finished

        let mut is_finished_offloading = HashSet::new();
        let mut completed_onboarding = HashSet::new();

        // before we process the maybes, add any newly annotated finished requests
        // to the maybe finished set
        for request_id in finished_requests {
            tracing::debug!(request_id, "marking request as finished");

            if !self.connector.has_slot(&request_id) {
                tracing::warn!(
                    request_id,
                    "finished request received for unknown request_id; assuming never started"
                );
                continue;
            }

            if self.maybe_finished_onboarding.contains(&request_id) {
                tracing::info!(
                    request_id,
                    "got a finished warning for a request that is onboarding"
                );
                // The request finished generating while its onboard is
                // still in flight — track it in the offloading set TOO, so
                // its completion is reported via finished_sending once the
                // slot completes. Without this, the request would never be
                // reported and vLLM would keep it forever under the
                // leader's always-return-true finish contract.
                self.maybe_finished_offloading.insert(request_id.clone());
            } else if self.maybe_finished_offloading.contains(&request_id) {
                tracing::warn!(
                    request_id,
                    "possibly got a duplicate finished request; request_id already in the maybe_finished_offloading set"
                );
            } else {
                tracing::debug!(
                    request_id,
                    "received finished request; adding to maybe_finished_offloading set"
                );
                self.maybe_finished_offloading.insert(request_id.clone());
            }
        }

        // visit each request slot in the maybe finished set
        for request_id in self.maybe_finished_offloading.iter() {
            if self.connector.has_slot(request_id) {
                if self.connector.is_complete(request_id) {
                    tracing::debug!(request_id, "request slot is finished");
                    is_finished_offloading.insert(request_id.clone());
                } else {
                    tracing::debug!(request_id, "request slot is not finished");
                }
            } else {
                // made this condition more strict slot existence checks were added as a prerequesite
                // to be added to the maybe_finished_offloading set.
                panic!(
                    "request slot missing for {request_id}; however, it was present when added to the maybe finished offloading set"
                );
            }
        }

        // remove the finished requests from the maybe finished set
        // note: when storing is finished we also remove the request from the engine state
        for request_id in &is_finished_offloading {
            self.maybe_finished_offloading.remove(request_id);

            // currently chomping the error as the engine is closed and we are shutting down
            if self.connector.has_slot(request_id) {
                self.connector.remove_slot(request_id);
            } else {
                tracing::debug!(
                    request_id,
                    "is_finished_offloading: request slot is not found - likely aborted, removing from is finished offloading set"
                );
            }
        }

        // visit each request slot in the maybe finished set to see if it is finished
        for request_id in self.maybe_finished_onboarding.iter() {
            if self.connector.has_slot(request_id) {
                if self.connector.is_complete(request_id) {
                    // The onboard (load) completed. Under kvbm's
                    // SYNCHRONOUS load contract (async_load=false: vLLM
                    // stream-ordered the transfers before the forward),
                    // vLLM does not consume the finished_recving channel
                    // for these — reporting a LIVE request there trips
                    // scheduler.py:_update_from_kv_xfer_finished's
                    // `assert RequestStatus.is_finished(...)` (the
                    // 2026-10-22 defect #4 fatal, which fired 0.7s after
                    // the first-ever completed H2D onboard). Reap the
                    // tracking entry only; KEEP the worker slot so the
                    // request's finish-time offload flow still finds it
                    // (removing it here would strand the request's
                    // finished_sending report and leak it in vLLM under
                    // the always-return-true finish contract).
                    tracing::debug!(
                        request_id,
                        "onboarding complete (sync contract: not reported via finished_recving; slot kept for the finish flow)"
                    );
                    completed_onboarding.insert(request_id.clone());
                } else {
                    tracing::debug!(request_id, "request slot is not finished");
                }
            } else {
                panic!(
                    "request slot missing for {request_id}; however, it was present when added to the maybe finished onboarding set"
                );
            }
        }

        // reap the completed onboarding entries from the maybe set
        for request_id in &completed_onboarding {
            self.maybe_finished_onboarding.remove(request_id);
        }

        // finished_recving stays empty under the sync load contract:
        // vLLM's recving path is only for WAITING_FOR_REMOTE_KVS
        // (async_load=true) or finished requests, and kvbm uses neither.
        (is_finished_offloading, HashSet::new())
    }
}

#[pyclass]
pub struct PyKvConnectorWorker {
    connector_worker: Box<dyn Worker>,
}

#[pymethods]
impl PyKvConnectorWorker {
    #[new]
    #[pyo3(signature = (py_drt, vllm_worker_id))]
    pub fn new(py_drt: Option<PyObject>, vllm_worker_id: String) -> PyResult<Self> {
        let drt: Option<Arc<DistributedRuntime>> = Python::with_gil(|py| {
            if let Some(obj) = py_drt {
                extract_distributed_runtime_from_obj(py, obj)
            } else {
                Ok(None)
            }
        })?;

        let connector_worker: Box<dyn Worker> =
            Box::new(KvConnectorWorker::new(drt, vllm_worker_id).map_err(to_pyerr)?);
        Ok(Self { connector_worker })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (num_device_blocks, page_size, device_id, dtype_width_bytes, kv_caches, raw_event_handles, device_layout_type=None, host_layout_type=None, disk_layout_type=None, outer_dim=None, inner_dim=None, group_of_layer=None, group_specs=None, active_save_layers=None))]
    pub fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: Vec<(String, Py<PyAny>)>,
        raw_event_handles: Vec<u64>,
        device_layout_type: Option<PyLayoutType>,
        host_layout_type: Option<PyLayoutType>,
        disk_layout_type: Option<PyLayoutType>,
        outer_dim: Option<usize>,
        inner_dim: Option<usize>,
        group_of_layer: Option<Vec<usize>>,
        group_specs: Option<Vec<(bool, Option<usize>)>>,
        active_save_layers: Option<usize>,
    ) -> PyResult<()> {
        // Convert Python tensors to Rust VllmTensor objects
        let mut rust_kv_caches = Vec::new();
        for (layer_name, py_tensor) in kv_caches {
            let vllm_tensor = Arc::new(VllmTensor::new(py_tensor).map_err(to_pyerr)?);
            rust_kv_caches.push((layer_name, vllm_tensor));
        }

        self.connector_worker
            .register_kv_caches(
                num_device_blocks,
                page_size,
                device_id,
                dtype_width_bytes,
                rust_kv_caches,
                raw_event_handles,
                device_layout_type.map(|py_layout| py_layout.into()),
                host_layout_type.map(|py_layout| py_layout.into()),
                disk_layout_type.map(|py_layout| py_layout.into()),
                outer_dim,
                inner_dim,
                group_of_layer,
                group_specs,
                active_save_layers,
            )
            .map_err(to_pyerr)
    }

    pub fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> PyResult<()> {
        self.connector_worker
            .bind_connector_metadata(metadata)
            .map_err(to_pyerr)
    }

    pub fn clear_connector_metadata(&mut self) {
        self.connector_worker.clear_connector_metadata()
    }

    pub fn save_kv_layer(&mut self, layer_name: String, _kv_layer: Py<PyAny>) -> PyResult<()> {
        // Note: kv_layer is not used in the current implementation
        self.connector_worker
            .save_kv_layer(layer_name)
            .map_err(to_pyerr)
    }

    /// Hybrid (HMA) post-forward save trigger. `event_handle` is a freshly
    /// recorded CUDA event (raw `cuda_event` handle) whose completion
    /// covers all registered layers; attention-only workers no-op.
    pub fn wait_for_save(&mut self, event_handle: u64) -> PyResult<()> {
        self.connector_worker
            .wait_for_save(event_handle)
            .map_err(to_pyerr)
    }

    pub fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>) {
        self.connector_worker.get_finished(finished_requests)
    }
}

use cudarc::driver::sys::{
    CUcontext, CUevent, cuCtxGetCurrent, cuEventSynchronize, cudaError_enum,
};
use std::ptr;

// todo(ryan): we will need this if we farm off the cuEventSynchronize to another thread
fn _get_current_context() -> CUcontext {
    let mut ctx: CUcontext = ptr::null_mut();
    let status = unsafe { cuCtxGetCurrent(&mut ctx) };
    assert_eq!(
        status,
        cudaError_enum::CUDA_SUCCESS,
        "cuCtxGetCurrent failed"
    );
    assert!(!ctx.is_null(), "Torch has not set a CUDA context");
    ctx
}

pub fn event_sync_blocking(event: u64) {
    let status = unsafe { cuEventSynchronize(event as CUevent) };
    assert_eq!(
        status,
        cudaError_enum::CUDA_SUCCESS,
        "cuEventSynchronize failed"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE ON TEST SCOPE: the save-trigger/wait_for_save/flush paths are
    // not unit-testable here. `KvConnectorWorker::new`'s debug-mode
    // landing pads instantiate the drop glue of
    // `kv_cache_layers: Vec<(String, Arc<VllmTensor>)>` — VllmTensor owns
    // a `Py<PyAny>`, so keeping `new` alive pulls `_Py_DecRef` into the
    // link, and the extension-module feature deliberately omits
    // libpython. The trigger's `event_sync_blocking` (cuEventSynchronize)
    // additionally needs a live CUDA context. Both paths are exercised
    // on the GPU node (Task 8; see the phase-2 plan doc
    // Research/2026-10-13-kvbm-hybrid-multigroup-phase2-plan.md). What IS
    // testable on CPU is the registration -> LayerGroup mapping below,
    // which is where the fp4's interleaved layout and the physical
    // mamba page_bytes must survive.

    #[test]
    fn build_layer_groups_none_inputs_keep_legacy_path() {
        assert!(
            build_layer_groups(None, None, 7, 16, 1, None, None)
                .unwrap()
                .is_empty()
        );
        assert!(
            build_layer_groups(Some(&[]), Some(&[]), 0, 16, 1, None, None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn build_layer_groups_interleaved_hybrid_fp4_pattern() {
        // The fp4 layout: attention tensors at positions 3 and 6 inside
        // the mamba span. The mamba page_bytes is the physical [B, C]
        // row size; the attention group passes None so its page tokens
        // fall back to the registration page_size.
        let group_of_layer = vec![0, 0, 0, 1, 0, 0, 1];
        let group_specs = vec![(true, Some(3_207_168usize)), (false, None)];
        let groups = build_layer_groups(
            Some(&group_of_layer),
            Some(&group_specs),
            7,
            16,
            1,
            None,
            None,
        )
        .unwrap();

        assert_eq!(groups.len(), 2);

        assert!(groups[0].is_mamba());
        assert_eq!(groups[0].num_layers, 5);
        assert_eq!(groups[0].page_bytes, 3_207_168);
        assert_eq!(groups[0].tensor_indices, vec![0, 1, 2, 4, 5]);
        assert_eq!(groups[0].bytes_per_block(), 5 * 3_207_168);

        assert!(!groups[1].is_mamba());
        assert_eq!(groups[1].num_layers, 2);
        assert_eq!(groups[1].page_bytes, 16);
        assert_eq!(groups[1].tensor_indices, vec![3, 6]);
        assert!(groups[1].bytes_per_block() > 0);
    }

    #[test]
    fn build_layer_groups_reject_mamba_without_page_bytes() {
        let group_of_layer = vec![0];
        let group_specs = vec![(true, None)];
        let err = build_layer_groups(
            Some(&group_of_layer),
            Some(&group_specs),
            1,
            16,
            1,
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("mamba layer group 0 requires page_bytes"),
            "expected mamba page_bytes error, got: {err}"
        );
    }

    #[test]
    fn build_layer_groups_reject_length_mismatch() {
        let group_of_layer = vec![0, 0];
        let group_specs = vec![(false, None)];
        let err = build_layer_groups(
            Some(&group_of_layer),
            Some(&group_specs),
            7,
            16,
            1,
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("2 entries but 7 tensors were registered"),
            "expected length-mismatch error, got: {err}"
        );
    }

    #[test]
    fn build_layer_groups_reject_unknown_group_id() {
        let group_of_layer = vec![0, 5];
        let group_specs = vec![(false, None)];
        let err = build_layer_groups(
            Some(&group_of_layer),
            Some(&group_specs),
            2,
            16,
            1,
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("group id 5 but only 1 groups"),
            "expected unknown-group error, got: {err}"
        );
    }

    #[test]
    fn build_layer_groups_spec_position_defines_group_identity() {
        // Tensor 0 belongs to group id 1 (mamba), tensor 1 to group id 0
        // (attention): the SPEC's position defines a group's kind, not
        // whichever id appears first in group_of_layer.
        let group_of_layer = vec![1, 0];
        let group_specs = vec![(false, None), (true, Some(96))];
        let groups = build_layer_groups(
            Some(&group_of_layer),
            Some(&group_specs),
            2,
            16,
            1,
            None,
            None,
        )
        .unwrap();

        assert_eq!(groups.len(), 2);
        assert!(!groups[0].is_mamba());
        assert_eq!(groups[0].tensor_indices, vec![1]);
        assert_eq!(groups[0].page_bytes, 16); // attention: page_size fallback
        assert!(groups[1].is_mamba());
        assert_eq!(groups[1].tensor_indices, vec![0]);
        assert_eq!(groups[1].page_bytes, 96);
    }

    #[test]
    fn build_layer_groups_unreferenced_spec_id_yields_empty_group() {
        // A spec id never referenced by group_of_layer builds an
        // empty-indices group; KvbmWorkerConfig::validate() is what
        // rejects it downstream, naming the group.
        let group_of_layer = vec![0, 0];
        let group_specs = vec![(false, None), (true, Some(96))];
        let groups = build_layer_groups(
            Some(&group_of_layer),
            Some(&group_specs),
            2,
            16,
            1,
            None,
            None,
        )
        .unwrap();

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].tensor_indices, vec![0, 1]);
        assert!(groups[1].tensor_indices.is_empty());

        let config = KvbmWorkerConfig::builder()
            .cancel_token(tokio_util::sync::CancellationToken::new())
            .num_device_blocks(4)
            .groups(groups)
            .build()
            .expect("config should build");
        let err = config.validate().unwrap_err().to_string();
        assert!(
            err.contains("LayerGroup 1 has no tensor indices"),
            "expected validate()'s empty-group rejection, got: {err}"
        );
    }

    #[test]
    fn build_layer_groups_empty_mamba_spec_without_page_bytes_names_emptiness() {
        // Without the emptiness check this would misreport the problem
        // as a missing page_bytes.
        let group_of_layer = vec![0, 0];
        let group_specs = vec![(false, None), (true, None)];
        let err = build_layer_groups(
            Some(&group_of_layer),
            Some(&group_specs),
            2,
            16,
            1,
            None,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("mamba layer group 1 has no layers in group_of_layer"),
            "expected empty-group error naming the group, got: {err}"
        );
        assert!(
            !err.contains("requires page_bytes"),
            "the emptiness error must fire before the page_bytes error: {err}"
        );
    }

    #[test]
    fn build_layer_groups_attention_uses_explicit_dims_when_given() {
        // MLA-style registration: explicit outer/inner dims are stored on
        // the attention group (derive_group_geometry consumes them via
        // the config-level explicit pair).
        let group_of_layer = vec![0, 0];
        let group_specs = vec![(false, None)];
        let groups = build_layer_groups(
            Some(&group_of_layer),
            Some(&group_specs),
            2,
            16,
            2,
            Some(1),
            Some(576),
        )
        .unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].outer_dim, 1);
        assert_eq!(groups[0].inner_dim, 576);
        assert_eq!(groups[0].dtype_width_bytes, 2);
        assert_eq!(groups[0].bytes_per_block(), 2 * 16 * 576 * 2);
    }
}

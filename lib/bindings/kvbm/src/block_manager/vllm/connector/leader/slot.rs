// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    any::Any,
    cmp::max,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use dynamo_llm::{
    block_manager::{
        BlockPool, NixlRegisterableStorage, Storage,
        block::{BlockMetadata, locality::LocalityProvider},
        config::should_bypass_cpu_cache,
        connector::protocol::{LeaderTransferRequest, RequestType, TransferType},
        distributed::{BlockTransferPool, BlockTransferRequest, KvbmLeader},
    },
    tokens::{SequenceHash, TokenBlock},
};
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tokio_util::sync::CancellationToken;

use crate::block_manager::cache_stats::CacheStatsTracker;
use crate::{get_current_cancel_token, get_current_tokio_handle};

use super::*;

#[derive(Debug, thiserror::Error)]
pub enum SlotError {
    #[error("slot not found")]
    NotFound,

    #[error("slot is in an invalid state: {0}")]
    InvalidState(String),

    #[error("slot operation failed: {0}")]
    InvalidOperation(String),

    #[error(transparent)]
    BlockPoolError(#[from] BlockPoolError),
}

pub trait SlotManager<R: RequestKey>: Send + Sync {
    type SlotType: Slot + ?Sized;

    fn has_slot(&self, request_id: &R) -> bool;

    /// Create a new slot for the given request ID, initial tokens and salt hash.
    fn create_slot(
        &self,
        request_id: &R,
        tokens: Vec<u32>,
        salt_hash: SaltHash,
    ) -> Result<(), SlotError>;

    fn get_slot(&self, request_id: &R) -> Result<Arc<Mutex<Self::SlotType>>, SlotError>;
    fn remove_slot(&self, request_id: &R) -> Result<(), SlotError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SlotState {
    /// The slot was not scheduled in the previous iteration.
    Initialized,

    /// The slot is prepared to load kv blocks from external storage; however, the onboarding operation
    /// has not been triggered yet. The usize is the number of tokens that are ready for onboarding.
    OnboardStaged(usize),

    /// The slot is actively copying blocks to device storage from some external storage(s).
    /// The usize is the number of tokens that are being onboarded.
    Onboarding(usize),

    /// The slot is actively prefilling the sequence.
    Prefilling,

    /// The slot is skipped prefill.
    SkippedPrefill,

    /// The slot is actively participating in a forward pass which will result in one more more tokens
    /// to be applied to the sequence.
    Decoding,

    /// The slot is skipped decoding.
    SkippedDecode,

    /// The slot is marked as finished, but not all resources have been released.
    Finishing,

    /// The slot is finished and all resources have been released.
    Finished,

    /// The slot is preempted and is waiting for the next iteration to resume.
    Preempted,
}

#[allow(dead_code)]
pub trait Slot: std::fmt::Debug {
    fn request_id(&self) -> &str;

    fn state(&self) -> SlotState;

    fn sequence(&self) -> &TokenBlockSequence;

    /// The number of tokens that have been computed on the device, i.e. the number of tokens for which we have ownership
    /// of computed kv blocks in the device storage.
    fn computed_tokens(&self) -> usize;

    fn apply_scheduler_output(
        &mut self,
        tokens: &[u32],
        block_ids: &[usize],
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
        priorities: Option<&[u32]>,
        external_sequence_hashes: Option<&[SequenceHash]>,
    ) -> Result<(), SlotError>;

    fn record_start_iteration(&mut self, iteration: u64) -> Result<(), SlotError>;

    fn mark_as_prefilling(&mut self, iteration: u64) -> Result<(), SlotError>;
    fn mark_as_decoding(&mut self, iteration: u64) -> Result<(), SlotError>;

    fn mark_as_finished(&mut self, iteration: u64) -> Result<(), SlotError>;

    /// The number of device blocks that have been allocated to the slot.
    fn num_device_blocks_allocated(&self) -> usize;

    /// Find all possible block matches for remaining known tokens in some local storage, i.e. look up and take ownership
    /// of any kv blocks for tokens in the isl that are not already in memory on the device, but on some local storage.
    ///
    /// If external tokens are matched, then the slot will transition to the [`SlotState::Onboarding`] state.
    /// `num_computed_tokens` is the number of tokens that have been computed on the device, this indicated the number of
    /// blocks in the ISL sequence that we should skip before we start looking for matches.
    fn acquire_local_matches(&mut self, num_computed_tokens: usize) -> Result<(), SlotError>;

    /// Trigger the onboarding operation for the slot.
    fn trigger_onboarding(&mut self, num_external_tokens: usize) -> Result<(), SlotError>;

    /// Take all pending operations for the slot.
    fn take_pending_operations(&mut self) -> Option<Vec<WorkerTransferRequest>>;

    /// Record the number of tokens that were cached on the device.
    fn record_cached_device_tokens(&mut self, num_tokens: usize);

    /// Record the number of tokens that were cached on the host.
    fn record_cached_host_tokens(&mut self, num_tokens: usize);

    /// Record the number of tokens that were cached on the disk.
    fn record_cached_disk_tokens(&mut self, num_tokens: usize);

    /// Reset the slot after preemption.
    fn reset_after_preemption(&mut self);

    /// Reset the slot.
    fn reset(&mut self);

    /// Get a mutable reference to the slot as a dynamic Any.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait ExternallyManagedDeviceSlot: Slot {
    /// Since we do not control the device pool, nor do we have insight in how the device pool is managed,
    /// we must accept external updates to the computed position.
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError>;

    /// Append the given block ids to the slot.
    ///
    /// The external device block manager has provided a set of mutable blocks to the slot.
    fn append_mutable_device_blocks(&mut self, block_ids: &[BlockId]) -> Result<(), SlotError>;
}

pub struct ConnectorSlotManager<R: RequestKey> {
    slots: Mutex<HashMap<R, Arc<Mutex<VllmConnectorSlot>>>>,
    block_manager: VllmBlockManager,
    /// use this to issue [`LocalTransferRequest`]s to the transfer engine
    xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,
    _transfer_engine_handle: Option<CriticalTaskExecutionHandle>,
    /// Cache statistics tracker
    cache_stats: Arc<CacheStatsTracker>,
    /// KVBM metrics for exposing cache hit rates
    #[allow(dead_code)]
    kvbm_metrics: KvbmMetrics,
    /// Minimum priority threshold for host offload filtering (read once at init)
    offload_min_priority: u32,
    /// HMA group count, captured from the leader ONCE at construction. All
    /// three constructors await the worker handshake first, so this is the
    /// post-handshake value; the SAME capture is handed to every slot (which
    /// records one metadata op per leg) and to the transfer engine (which
    /// sends one wire leg per group), keeping the two sides in lockstep.
    num_groups: usize,
}

impl std::fmt::Debug for ConnectorSlotManager<String> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectorSlotManager").finish()
    }
}

impl<R: RequestKey> ConnectorSlotManager<R> {
    pub fn new(
        block_manager: VllmBlockManager,
        leader: Arc<KvbmLeader>,
        kvbm_metrics: KvbmMetrics,
        identifier: Option<String>,
    ) -> Self {
        let cache_stats = Arc::new(CacheStatsTracker::new(identifier));
        let offload_min_priority = std::env::var("DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let kvbm_metrics_clone = kvbm_metrics.clone();
        let cache_stats_clone = cache_stats.clone();

        // Spawn a background task to periodically update metrics and log cache hit rates
        let handle = get_current_tokio_handle();
        handle.spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
            loop {
                interval.tick().await;
                // Update Prometheus metrics
                let host_rate = cache_stats_clone.host_hit_rate();
                let disk_rate = cache_stats_clone.disk_hit_rate();
                kvbm_metrics_clone.update_cache_hit_rates(host_rate, disk_rate, 0.0);
                // Also log cache hit rates periodically
                cache_stats_clone.maybe_log();
            }
        });
        tracing::debug!(
            "creating slot manager with block size: {}",
            block_manager.block_size()
        );

        let (xfer_tx, xfer_rx) = mpsc::unbounded_channel();

        // Post-handshake capture (see the field doc): one shared group count
        // for slots and the transfer engine.
        let num_groups = leader.num_groups();

        let mut xfer_engine =
            LocalTransferEngine::new(block_manager.clone(), leader, num_groups, xfer_rx);
        let primary_token = get_current_cancel_token();
        let primary_token_clone = primary_token.clone();
        let runtime_primary = get_current_tokio_handle();
        let runtime_primary_clone = runtime_primary.clone();
        let kvbm_metrics_clone = kvbm_metrics.clone();

        let xfer_engine_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token| async move {
                xfer_engine
                    .execute(
                        cancellation_token,
                        runtime_primary_clone,
                        primary_token_clone,
                        kvbm_metrics_clone,
                    )
                    .await
            },
            primary_token,
            "LocalTransferEngine",
            &runtime_primary,
        )
        .unwrap();

        Self {
            slots: Mutex::new(HashMap::new()),
            block_manager,
            xfer_tx,
            _transfer_engine_handle: Some(xfer_engine_task),
            cache_stats,
            kvbm_metrics: kvbm_metrics.clone(),
            offload_min_priority,
            num_groups,
        }
    }

    pub fn reset_prefix_cache(&self) -> Result<(), SlotError> {
        if let Some(disk) = self.block_manager.disk() {
            disk.reset_blocking()?;
            tracing::debug!("reset disk prefix cache");
        }

        if let Some(host) = self.block_manager.host() {
            host.reset_blocking()?;
            tracing::debug!("reset host prefix cache");
        }

        if let Some(device) = self.block_manager.device() {
            device.reset_blocking()?;
            tracing::debug!("reset device prefix cache");
        }

        self.slots.lock().unwrap().clear();
        Ok(())
    }
}

impl<R: RequestKey> SlotManager<R> for ConnectorSlotManager<R> {
    type SlotType = dyn ExternallyManagedDeviceSlot;

    fn has_slot(&self, request_id: &R) -> bool {
        self.slots.lock().unwrap().contains_key(request_id)
    }

    fn create_slot(
        &self,
        request_id: &R,
        tokens: Vec<u32>,
        salt_hash: SaltHash,
    ) -> Result<(), SlotError> {
        tracing::debug!(
            "creating slot with request_id: {}, num_tokens: {}",
            request_id,
            tokens.len()
        );
        let slot = VllmConnectorSlot::new(
            request_id.to_string(),
            tokens.into(),
            salt_hash,
            self.block_manager.clone(),
            self.xfer_tx.clone(),
            self.cache_stats.clone(),
            self.offload_min_priority,
            self.num_groups,
        );
        self.slots
            .lock()
            .unwrap()
            .insert(request_id.clone(), Arc::new(Mutex::new(slot)));
        Ok(())
    }

    fn get_slot(&self, request_id: &R) -> Result<Arc<Mutex<Self::SlotType>>, SlotError> {
        let slots = self.slots.lock().unwrap();
        let slot = slots.get(request_id).ok_or(SlotError::NotFound)?;
        Ok(slot.clone())
    }

    fn remove_slot(&self, request_id: &R) -> Result<(), SlotError> {
        self.slots.lock().unwrap().remove(request_id);
        Ok(())
    }
}

impl<R: RequestKey> Drop for ConnectorSlotManager<R> {
    fn drop(&mut self) {
        if let Some(task) = self._transfer_engine_handle.take() {
            task.cancel();
            task.detach();
        }
    }
}

pub struct VllmConnectorSlot {
    request_id: String,

    /// The state of the slot.
    state: SlotState,

    // /// Current position in the sequence of tokens that have been computed.
    // /// When the slot is initialized, we populate the sequence with the prefill tokens.
    // /// However, those tokens are not yet prefilled, so they are not yet represented
    // /// in the sequence_position.
    // computed_position: usize,
    /// The sequence of token blocks
    sequence: TokenBlockSequence,

    /// The mutable blocks id (device)
    device_blocks: Vec<BlockId>,

    /// Blocks to be onboarded from the host
    /// We must hold these blocks in the slot state until the scheduler trigger the onboarding.
    staging_from_host: Option<Vec<ImmutableBlock<DevbarStorage, VllmLocality, BasicMetadata>>>,

    /// Blocks to be onboarded from the disk
    /// We must hold these blocks in the slot state until the scheduler trigger the onboarding.
    staging_from_disk: Option<Vec<ImmutableBlock<DiskStorage, VllmLocality, BasicMetadata>>>,

    /// The number of blocks cached from the device
    tokens_cached_from_device: usize,

    /// The number of blocks cached from the host
    tokens_cached_from_host: usize,

    /// The number of blocks cached from the disk
    tokens_cached_from_disk: usize,

    /// Block manager for device pool operations (cache lookup, onboarding).
    /// `None` only in unit tests where these operations are not exercised.
    block_manager: Option<VllmBlockManager>,

    block_size: usize,

    iteration_first_scheduled: Option<u64>,

    pending_operations: Option<Vec<WorkerTransferRequest>>,

    /// use this to issue [`LocalTransferRequest`]s to the transfer engine
    xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,

    /// This is the current position for which we are applying some number of active/scheduled tokens.
    /// On application, then we decide what actions we take.
    /// This the point that we will call our generic policy object.
    current_position: usize,

    /// The number of blocks that have been evaluated by the policy.
    /// Each policy evaluation will skip the already evaluated blocks.
    evaluated_blocks: usize,

    /// Whether we actually performed a cache lookup for this request
    performed_cache_lookup: bool,

    /// Total number of blocks queried from host/disk cache
    total_blocks_queried: usize,

    /// Cache statistics tracker for this KVBM instance
    cache_stats: Arc<CacheStatsTracker>,

    /// Minimum priority threshold for offload filtering.
    /// All blocks after the first occurance of block priority < threshold are not offloaded.
    offload_min_priority: u32,

    /// Block index where offload was terminated due to priority filtering.
    /// When Some, no further blocks will be offloaded to ensure global contiguity.
    offload_terminated_at_block: Option<usize>,

    /// Stored block priorities from previous apply_scheduler_output calls.
    /// Used as fallback when priorities=None in subsequent chunked prefill iterations.
    stored_block_priorities: HashMap<BlockId, u32>,

    /// HMA group count captured by the slot manager from the leader's
    /// handshake state at slot-manager construction (post-handshake; the
    /// same capture the transfer engine uses). Each offload/onboard op
    /// records ONE metadata operation per group so the worker-side
    /// rendezvous — one Worker + one Transfer per `(request_id, uuid)` —
    /// matches every wire leg.
    num_groups: usize,
}

impl VllmConnectorSlot {
    #[allow(clippy::too_many_arguments)]
    fn new(
        request_id: String,
        tokens: Tokens,
        salt_hash: SaltHash,
        block_manager: VllmBlockManager,
        xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,
        cache_stats: Arc<CacheStatsTracker>,
        offload_min_priority: u32,
        num_groups: usize,
    ) -> Self {
        assert!(!tokens.is_empty(), "tokens must be non-empty");
        let block_size = block_manager.block_size();
        debug_assert!(block_size.is_power_of_two() && block_size <= 1024);
        let sequence = TokenBlockSequence::new(tokens, block_size as u32, Some(salt_hash));

        Self {
            request_id,
            sequence,
            block_manager: Some(block_manager),
            block_size,
            xfer_tx,
            // default values
            state: SlotState::Initialized,
            iteration_first_scheduled: None,
            current_position: 0,
            evaluated_blocks: 0,
            device_blocks: Vec::new(),
            staging_from_host: None,
            staging_from_disk: None,
            pending_operations: None,
            tokens_cached_from_device: 0,
            tokens_cached_from_host: 0,
            tokens_cached_from_disk: 0,
            performed_cache_lookup: false,
            total_blocks_queried: 0,
            cache_stats,
            offload_min_priority,
            offload_terminated_at_block: None,
            stored_block_priorities: HashMap::new(),
            num_groups: num_groups.max(1),
        }
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn new_for_test(
        request_id: String,
        tokens: Tokens,
        salt_hash: SaltHash,
        block_size: usize,
        xfer_tx: mpsc::UnboundedSender<LocalTransferRequest>,
        cache_stats: Arc<CacheStatsTracker>,
        offload_min_priority: u32,
        num_groups: usize,
    ) -> Self {
        assert!(!tokens.is_empty(), "tokens must be non-empty");
        let sequence = TokenBlockSequence::new(tokens, block_size as u32, Some(salt_hash));
        Self {
            request_id,
            sequence,
            block_manager: None,
            block_size,
            xfer_tx,
            state: SlotState::Initialized,
            iteration_first_scheduled: None,
            current_position: 0,
            evaluated_blocks: 0,
            device_blocks: Vec::new(),
            staging_from_host: None,
            staging_from_disk: None,
            pending_operations: None,
            tokens_cached_from_device: 0,
            tokens_cached_from_host: 0,
            tokens_cached_from_disk: 0,
            performed_cache_lookup: false,
            total_blocks_queried: 0,
            cache_stats,
            offload_min_priority,
            offload_terminated_at_block: None,
            stored_block_priorities: HashMap::new(),
            num_groups: num_groups.max(1),
        }
    }

    #[cfg(test)]
    fn device_blocks_snapshot(&self) -> &[BlockId] {
        &self.device_blocks
    }

    fn sync_external_sequence_hashes(
        &mut self,
        external_sequence_hashes: &[SequenceHash],
    ) -> Result<(), SlotError> {
        assert_eq!(
            external_sequence_hashes.len(),
            self.sequence.blocks().len(),
            "external_sequence_hashes length ({}) must match completed block count ({}) for request {}",
            external_sequence_hashes.len(),
            self.sequence.blocks().len(),
            self.request_id
        );

        self.sequence
            .sync_external_sequence_hashes(external_sequence_hashes);

        for block in self.sequence.blocks() {
            block.assert_external_hashes_assigned();
        }

        Ok(())
    }

    fn mark_as_skipped_prefill(&mut self) -> Result<(), SlotError> {
        if self.state != SlotState::Prefilling {
            return Err(SlotError::InvalidState(format!(
                "cannot mark slot as skipped prefill in state {:?}",
                self.state
            )));
        }
        self.state = SlotState::SkippedPrefill;
        Ok(())
    }

    fn mark_as_skipped_decode(&mut self) -> Result<(), SlotError> {
        if self.state != SlotState::Decoding {
            return Err(SlotError::InvalidState(format!(
                "cannot mark slot as skipped decode in state {:?}",
                self.state
            )));
        }
        self.state = SlotState::SkippedDecode;
        Ok(())
    }

    pub fn mark_as_skipped(&mut self) -> Result<(), SlotError> {
        match self.state {
            SlotState::Prefilling => self.mark_as_skipped_prefill(),
            SlotState::Decoding => self.mark_as_skipped_decode(),
            SlotState::SkippedPrefill => Ok(()), // already skipped
            SlotState::SkippedDecode => Ok(()),  // already skipped
            _ => {
                tracing::debug!(
                    "slot is in the {:?} state; will not explicitly mark as skipped, request_id: {}",
                    self.state,
                    self.request_id
                );
                Ok(())
            }
        }
    }
}

impl std::fmt::Debug for VllmConnectorSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VllmConnectorSlot")
            .field("state", &self.state)
            .field("current_position", &self.current_position)
            .field("num_tokens", &self.sequence.total_tokens())
            .finish()
    }
}

impl Slot for VllmConnectorSlot {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn state(&self) -> SlotState {
        self.state
    }

    fn reset_after_preemption(&mut self) {
        assert!(self.staging_from_disk.is_none());
        assert!(self.staging_from_host.is_none());
        assert!(self.pending_operations.is_none());

        self.state = SlotState::Preempted;
        self.iteration_first_scheduled = None;
        self.current_position = 0;
        self.evaluated_blocks = 0;
        self.device_blocks.clear();
        self.tokens_cached_from_device = 0;
        self.tokens_cached_from_host = 0;
        self.tokens_cached_from_disk = 0;
        self.performed_cache_lookup = false;
        self.total_blocks_queried = 0;
        self.offload_terminated_at_block = None;
        self.stored_block_priorities.clear();
    }

    fn reset(&mut self) {
        self.reset_after_preemption();
        self.state = SlotState::Initialized;
    }

    fn mark_as_prefilling(&mut self, _iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Prefilling;
        Ok(())
    }

    fn mark_as_decoding(&mut self, _iteration: u64) -> Result<(), SlotError> {
        self.state = SlotState::Decoding;
        Ok(())
    }

    fn record_cached_device_tokens(&mut self, num_tokens: usize) {
        self.tokens_cached_from_device = num_tokens;
        tracing::debug!("recording {} cached device tokens", num_tokens,);
    }

    fn record_cached_host_tokens(&mut self, num_tokens: usize) {
        self.tokens_cached_from_host = num_tokens;
        tracing::debug!("recording {} cached host tokens", num_tokens);
    }

    fn record_cached_disk_tokens(&mut self, num_tokens: usize) {
        self.tokens_cached_from_disk = num_tokens;
        tracing::debug!("recording {} cached disk tokens", num_tokens);
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = self.request_id.as_str()))]
    fn apply_scheduler_output(
        &mut self,
        tokens: &[u32],
        block_ids: &[BlockId],
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
        priorities: Option<&[u32]>,
        external_sequence_hashes: Option<&[SequenceHash]>,
    ) -> Result<(), SlotError> {
        tracing::debug!(
            "ENTRY: apply_scheduler_output: req={}, tokens.len={}, block_ids.len={}, computed={}, scheduled={}, \
             has_priorities={}, current_pos={}, evaluated_blocks={}, device_blocks_len={}",
            self.request_id,
            tokens.len(),
            block_ids.len(),
            num_computed_tokens,
            num_scheduled_tokens,
            priorities.is_some(),
            self.current_position,
            self.evaluated_blocks,
            self.device_blocks.len()
        );

        // Validate contract: priorities must match block_ids length when provided
        if let Some(prios) = priorities {
            assert_eq!(
                prios.len(),
                block_ids.len(),
                "priorities length ({}) must match block_ids length ({})",
                prios.len(),
                block_ids.len()
            );
        }

        if !tokens.is_empty() {
            tracing::debug!(
                "appending {} newly decoded tokens to sequence",
                tokens.len()
            );
            self.state = SlotState::Decoding;
            self.sequence.extend(tokens.into()).unwrap();
        } else {
            self.state = SlotState::Prefilling;
        }

        if let Some(external_sequence_hashes) = external_sequence_hashes {
            self.sync_external_sequence_hashes(external_sequence_hashes)?;
        }

        // Use max to advance both current_position and evaluated_blocks at least by num_computed_tokens.
        // This logic is to prevent redundant block offloading.
        self.current_position = max(self.current_position, num_computed_tokens);
        self.evaluated_blocks = max(self.evaluated_blocks, num_computed_tokens / self.block_size);

        // Apply new block_ids with suffix/prefix overlap contract.
        // Block IDs are unique, so we use an O(N) algorithm:
        //   1. Locate block_ids[0] in device_blocks via rposition (at most one match).
        //   2. Verify device_blocks[pos..] == block_ids[..suffix_len] (the overlap).
        //   3. Extend with only the non-overlapping tail block_ids[overlap_len..].
        //
        // Valid cases:
        //   [3,4,5] + [6,7]       → [3,4,5,6,7]       (no overlap)
        //   [3,4,5,6,7] + [6,7,8,9] → [3,4,5,6,7,8,9] (suffix/prefix overlap of 2)
        //   [3,4,5] + [3,4,5]     → [3,4,5]            (full overlap)
        //
        // Invalid (panics):
        //   [3,4,5,6,7] + [6,8,9] → block 6 found but suffix doesn't match prefix
        if !block_ids.is_empty() {
            let overlap_len = if let Some(pos) = self
                .device_blocks
                .iter()
                .rposition(|&id| id == block_ids[0])
            {
                let suffix_len = self.device_blocks.len() - pos;
                assert!(
                    suffix_len <= block_ids.len()
                        && self.device_blocks[pos..] == block_ids[..suffix_len],
                    "device_blocks contract violation: block_ids[0]={} found at \
                     device_blocks[{}] but device_blocks[{}..] != block_ids[..{}] \
                     (device_blocks={:?}, block_ids={:?})",
                    block_ids[0],
                    pos,
                    pos,
                    suffix_len,
                    self.device_blocks,
                    block_ids
                );
                suffix_len
            } else {
                0
            };

            let new_ids = &block_ids[overlap_len..];

            if !new_ids.is_empty() {
                // Validate: no block in the non-overlapping portion should already exist.
                let existing: HashSet<BlockId> = self.device_blocks.iter().copied().collect();
                for id in new_ids {
                    assert!(
                        !existing.contains(id),
                        "device_blocks contract violation: block {} already in device_blocks \
                         but not part of suffix/prefix overlap (overlap_len={}, \
                         device_blocks={:?}, block_ids={:?})",
                        id,
                        overlap_len,
                        self.device_blocks,
                        block_ids
                    );
                }
                self.device_blocks.extend_from_slice(new_ids);
            }

            if overlap_len > 0 {
                tracing::debug!(
                    "DEDUP: suffix/prefix overlap of {} block_ids for req={}, extended with {} new",
                    overlap_len,
                    self.request_id,
                    new_ids.len()
                );
            }
        }

        // Store block→priority mapping for use in subsequent chunked prefill iterations.
        // In chunked prefill, new_request (chunk 1) carries priorities for ALL blocks,
        // but cached_request (chunk 2+) has priorities=None. Storing here lets us look up
        // priorities for blocks evaluated in later chunks.
        if let Some(prios) = priorities {
            for (block_id, priority) in block_ids.iter().zip(prios.iter()) {
                self.stored_block_priorities.insert(*block_id, *priority);
            }
        }

        // Early exit if offload has been permanently terminated.
        // This ensures global contiguity: once a gap is created by priority filtering,
        // no subsequent blocks will be offloaded for this request.
        if let Some(terminated_at) = self.offload_terminated_at_block {
            tracing::debug!(
                "offload terminated at block {}; skipping offload evaluation",
                terminated_at
            );
            self.current_position += num_scheduled_tokens;
            return Ok(());
        }

        // we should have enough device blocks to cover the newly scheduled tokens
        let next_position = self.current_position + num_scheduled_tokens;
        assert!(
            next_position <= self.device_blocks.len() * self.block_size,
            "next_position: {} > device_blocks.len() {} * block_size {}",
            next_position,
            self.device_blocks.len(),
            self.block_size
        );

        if next_position > self.sequence.total_tokens() {
            // vllm stopped providing tokens, so we are done
            self.state = SlotState::Decoding;
            tracing::debug!(
                "connector source stopped providing tokens; no further evaluation possible"
            );
            return Ok(());
        }

        // now we decide what we should do from the current position to the num_scheduled_tokens
        tracing::debug!(
            "applying kv cache policy at current_position: {}; num_scheduled_tokens: {}; num_evaluated_blocks: {}",
            self.current_position,
            num_scheduled_tokens,
            self.evaluated_blocks
        );

        // TODO(ryan) - apply policy
        let next_position = self.current_position + num_scheduled_tokens;

        debug_assert!(next_position / self.block_size >= self.evaluated_blocks);

        let num_candidate_blocks = (next_position / self.block_size) - self.evaluated_blocks;

        tracing::debug!(
            "evaluating policy with the following parameters: state: {:?}; current_position: {}; num_candidate_blocks: {}; num_scheduled_tokens: {}",
            self.state,
            self.current_position,
            num_candidate_blocks,
            num_scheduled_tokens
        );

        if num_candidate_blocks != 0 {
            // Get candidate block IDs
            let candidate_block_ids: Vec<usize> = self
                .device_blocks
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_candidate_blocks)
                .copied()
                .collect();

            // Look up candidate priorities from stored_block_priorities.
            // The HashMap is populated above (lines 700-706) for every block_id
            // that arrives with priorities. This replaces fragile offset-based indexing
            // that assumed block_ids was appended verbatim to device_blocks.
            let candidate_priorities: Vec<u32> = candidate_block_ids
                .iter()
                .map(|id| self.stored_block_priorities.get(id).copied().unwrap_or(0))
                .collect();

            assert_eq!(
                candidate_block_ids.len(),
                num_candidate_blocks,
                "device block overflow - candidate blocks exceed block count at offset {}",
                self.evaluated_blocks
            );

            // Apply contiguous priority filtering: find how many blocks from the start
            // meet the minimum priority threshold. Stop at first block below threshold.
            let num_blocks_to_offload = if self.offload_min_priority > 0 {
                candidate_priorities
                    .iter()
                    .take_while(|&&priority| priority >= self.offload_min_priority)
                    .count()
            } else {
                num_candidate_blocks
            };

            tracing::debug!(
                "OFFLOAD_DECISION: req={}, num_candidate={}, num_to_offload={}, threshold={}, \
                 candidate_block_ids={:?}, candidate_priorities={:?}",
                self.request_id,
                num_candidate_blocks,
                num_blocks_to_offload,
                self.offload_min_priority,
                &candidate_block_ids,
                &candidate_priorities
            );

            if num_blocks_to_offload > 0 {
                if self.offload_min_priority > 0 {
                    tracing::debug!(
                        "priority filtering: offloading {}/{} blocks (threshold={})",
                        num_blocks_to_offload,
                        num_candidate_blocks,
                        self.offload_min_priority
                    );
                }

                let offload_block_ids: Vec<usize> = candidate_block_ids
                    .into_iter()
                    .take(num_blocks_to_offload)
                    .collect();

                let offload_token_blocks: Vec<TokenBlock> = self
                    .sequence
                    .blocks()
                    .iter()
                    .skip(self.evaluated_blocks)
                    .take(num_blocks_to_offload)
                    .cloned()
                    .collect();

                let offload_priorities: Vec<u32> = candidate_priorities
                    .iter()
                    .take(num_blocks_to_offload)
                    .copied()
                    .collect();

                if external_sequence_hashes.is_some() {
                    for block in &offload_token_blocks {
                        block.assert_external_hashes_assigned();
                    }
                }

                self.offload_blocks(
                    &offload_block_ids,
                    &offload_token_blocks,
                    &offload_priorities,
                )
                .expect("failed to offload blocks");
            } else if self.offload_min_priority > 0 {
                tracing::debug!(
                    "priority filtering: skipping all {} candidate blocks (threshold={})",
                    num_candidate_blocks,
                    self.offload_min_priority
                );
            }

            // Check if we skipped any blocks due to priority filtering.
            // If so, terminate offloading for this request to ensure global contiguity.
            if num_blocks_to_offload < num_candidate_blocks {
                let termination_index = self.evaluated_blocks + num_blocks_to_offload;
                self.offload_terminated_at_block = Some(termination_index);

                tracing::info!(
                    request_id = %self.request_id,
                    "offload terminated at block {}: priority {} < threshold {}; \
                     no further blocks will be offloaded",
                    termination_index,
                    candidate_priorities.get(num_blocks_to_offload).copied().unwrap_or(0),
                    self.offload_min_priority
                );
            }

            self.evaluated_blocks += num_candidate_blocks;
        }

        // done applying policy
        tracing::debug!(
            "done applying kv cache policy at current_position: {}; num_scheduled_tokens: {}",
            self.current_position,
            num_scheduled_tokens
        );

        // advance current and computed position
        self.current_position += num_scheduled_tokens;

        Ok(())
    }

    fn record_start_iteration(&mut self, iteration: u64) -> Result<(), SlotError> {
        if self.iteration_first_scheduled.is_none() {
            self.iteration_first_scheduled = Some(iteration);
        }
        Ok(())
    }

    fn mark_as_finished(&mut self, _iteration: u64) -> Result<(), SlotError> {
        // Report cache statistics if we performed a cache lookup
        if self.performed_cache_lookup {
            let block_size = self.block_size;

            // Convert cached tokens to blocks (rounding up)
            let host_blocks = self.tokens_cached_from_host.div_ceil(block_size);
            let disk_blocks = self.tokens_cached_from_disk.div_ceil(block_size);

            tracing::debug!(
                request_id = %self.request_id,
                "Reporting cache stats: host_blocks={}, disk_blocks={}, total_blocks_queried={}, tokens_from_host={}, tokens_from_disk={}",
                host_blocks,
                disk_blocks,
                self.total_blocks_queried,
                self.tokens_cached_from_host,
                self.tokens_cached_from_disk
            );

            self.cache_stats
                .record(host_blocks, disk_blocks, self.total_blocks_queried);
        }

        // Check if there are any pending operations
        let has_pending_ops = self
            .pending_operations
            .as_ref()
            .map(|ops| !ops.is_empty())
            .unwrap_or(false);

        if has_pending_ops {
            // There are pending operations - need to wait for them to complete
            self.state = SlotState::Finishing;
            tracing::debug!(
                request_id = %self.request_id,
                pending_operations = self.pending_operations.as_ref().unwrap().len(),
                "request set to finish (with pending operations): cached_gpu_tokens: {}; cached_host_tokens: {}; cached_disk_tokens: {}",
                self.tokens_cached_from_device,
                self.tokens_cached_from_host,
                self.tokens_cached_from_disk
            );
        } else {
            // No pending operations - can immediately mark as finished
            self.state = SlotState::Finished;
            tracing::debug!(
                request_id = %self.request_id,
                "request set to finished (no pending operations): cached_gpu_tokens: {}; cached_host_tokens: {}; cached_disk_tokens: {}",
                self.tokens_cached_from_device,
                self.tokens_cached_from_host,
                self.tokens_cached_from_disk
            );
        }
        Ok(())
    }

    fn sequence(&self) -> &TokenBlockSequence {
        &self.sequence
    }

    fn computed_tokens(&self) -> usize {
        self.current_position
    }

    fn num_device_blocks_allocated(&self) -> usize {
        self.device_blocks.len()
    }

    fn take_pending_operations(&mut self) -> Option<Vec<WorkerTransferRequest>> {
        self.pending_operations.take()
    }

    #[tracing::instrument(level = "debug", skip_all)]
    fn acquire_local_matches(&mut self, num_computed_tokens: usize) -> Result<(), SlotError> {
        if matches!(self.state(), SlotState::OnboardStaged(_)) {
            tracing::debug!("slot is already in the OnboardStaged state; skipping lookup");
            return Ok(());
        }

        if !matches!(self.state(), SlotState::Initialized | SlotState::Preempted) {
            return Err(SlotError::InvalidOperation(format!(
                "slot must be in the NotScheduled or Preempted state to acquire local matches; got {:?}",
                self.state()
            )));
        }

        if matches!(self.state(), SlotState::Preempted) {
            tracing::info!("slot is in the Preempted state; we get another chance to match");
        }

        let block_size = self.block_size;
        let num_computed_blocks = num_computed_tokens / block_size;
        debug_assert!(num_computed_tokens.is_multiple_of(block_size));

        let sequence_hashes = self
            .sequence()
            .blocks()
            .iter()
            .map(|b| b.sequence_hash())
            .collect::<Vec<_>>();

        // we start matching non-device blocks after the device blocks
        let search_offset = num_computed_blocks;

        // Calculate how many blocks we're querying from host/disk
        let blocks_to_lookup = &sequence_hashes[search_offset..];

        tracing::debug!("matching against {} block hashes", blocks_to_lookup.len());

        // If there are no blocks to lookup (GPU has everything), return early
        if blocks_to_lookup.is_empty() {
            tracing::debug!(
                request_id = %self.request_id,
                "no blocks to lookup from host/disk; GPU has all blocks"
            );
            // Still mark that we performed a lookup (even though we didn't need to query)
            self.performed_cache_lookup = true;
            self.total_blocks_queried = 0;
            return Ok(());
        }

        // Mark that we're performing a cache lookup and track the total blocks
        self.performed_cache_lookup = true;
        self.total_blocks_queried = blocks_to_lookup.len();

        tracing::debug!(
            request_id = %self.request_id,
            "Starting cache lookup: querying {} blocks from host/disk (num_computed_blocks={})",
            blocks_to_lookup.len(),
            num_computed_blocks
        );

        // we should do this opportunistically after this operation is done
        // ideally it was triggered by the match_sequence_hashes_blocking calls directly

        // if let Some(host) = self.block_manager.host() {
        //     host.touch_blocks_blocking(&sequence_hashes)?;
        // }

        // if let Some(disk) = self.block_manager.disk() {
        //     disk.touch_blocks_blocking(&sequence_hashes)?;
        // }

        let mut host_blocks = self
            .block_manager
            .as_ref()
            .and_then(|bm| bm.host())
            .map(|host| host.match_sequence_hashes_blocking(blocks_to_lookup))
            .transpose()?
            .unwrap_or_default();

        let num_matched_host_blocks = host_blocks.len();
        self.record_cached_host_tokens(num_matched_host_blocks * block_size);

        // advance the search offset by the number of matched host blocks
        let search_offset = search_offset + num_matched_host_blocks;

        // start at host offset
        let mut disk_blocks = self
            .block_manager
            .as_ref()
            .and_then(|bm| bm.disk())
            .map(|disk| disk.match_sequence_hashes_blocking(&sequence_hashes[search_offset..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_disk_blocks = disk_blocks.len();
        self.record_cached_disk_tokens(num_matched_disk_blocks * block_size);

        let num_matched_blocks = num_matched_host_blocks + num_matched_disk_blocks;

        tracing::debug!(
            "successfully matched {} host blocks and {} disk blocks; {} total blocks",
            num_matched_host_blocks,
            num_matched_disk_blocks,
            num_matched_blocks
        );

        // early exit if we did not match any blocks
        if num_matched_blocks == 0 {
            return Ok(());
        }

        let mut num_new_matched_tokens = num_matched_blocks * block_size;

        // we are on a block boundary, so we need to throw away the last block
        if (num_computed_tokens + num_new_matched_tokens) == self.sequence().total_tokens() {
            tracing::debug!("on a block boundary, throwing away the last block");

            // we should have matched at least one block
            assert!(!host_blocks.is_empty() || !disk_blocks.is_empty());

            // pop from disk, or if there are none, then from host
            if disk_blocks.is_empty() {
                host_blocks.pop();
            } else {
                disk_blocks.pop();
            }

            // decrement the number of new matched tokens by the block size
            num_new_matched_tokens -= block_size;
        }

        // early exit if we need to onboard 0 blocks (after potentially dropping the last block)
        if num_new_matched_tokens == 0 {
            return Ok(());
        }

        self.staging_from_host = if !host_blocks.is_empty() {
            Some(host_blocks)
        } else {
            None
        };
        self.staging_from_disk = if !disk_blocks.is_empty() {
            Some(disk_blocks)
        } else {
            None
        };

        self.state = SlotState::OnboardStaged(num_new_matched_tokens);

        Ok(())
    }

    fn trigger_onboarding(&mut self, num_external_tokens: usize) -> Result<(), SlotError> {
        if !matches!(self.state(), SlotState::OnboardStaged(_)) {
            return Err(SlotError::InvalidOperation(format!(
                "slot must be in the OnboardStaged state to trigger onboarding; got {:?}",
                self.state()
            )));
        }

        debug_assert_eq!(self.evaluated_blocks, 0);
        debug_assert_eq!(self.current_position % self.block_size, 0);
        debug_assert_eq!(num_external_tokens % self.block_size, 0);

        let num_computed_blocks = self.current_position / self.block_size;

        // shift the evaluated blocks position to the end of the computed/cached blocks
        self.evaluated_blocks = num_computed_blocks;

        // match the host / disk blocks to the newly assigned mutable device blocks
        if let Some(host_blocks) = self.staging_from_host.take() {
            let num_host_blocks = host_blocks.len();

            // get device block ids
            let dst_block_ids = self
                .device_blocks
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_host_blocks)
                .copied()
                .collect::<Vec<_>>();

            debug_assert_eq!(dst_block_ids.len(), num_host_blocks);

            // construct offload requests - transfer engine + worker
            let src_blocks = Box::new(AnyImmutableBlocks::<DevbarStorage, _, _>::new(host_blocks));

            self.onboard_blocks(src_blocks, dst_block_ids)?;

            // shift the evaluated blocks position to the end of the computed/cached blocks
            self.evaluated_blocks += num_host_blocks;
        }

        if let Some(disk_blocks) = self.staging_from_disk.take() {
            let num_disk_blocks = disk_blocks.len();

            // get device block ids
            let dst_block_ids = self
                .device_blocks
                .iter()
                .skip(self.evaluated_blocks)
                .take(num_disk_blocks)
                .copied()
                .collect::<Vec<_>>();

            debug_assert_eq!(dst_block_ids.len(), num_disk_blocks);

            // construct offload requests - transfer engine + worker
            let src_blocks = Box::new(AnyImmutableBlocks::<DiskStorage, _, _>::new(disk_blocks));

            self.onboard_blocks(src_blocks, dst_block_ids)?;

            // shift the evaluated blocks position to the end of the computed/cached blocks
            self.evaluated_blocks += num_disk_blocks;
        }

        self.state = SlotState::Onboarding(num_external_tokens);
        self.advance_computed_position(num_external_tokens)?;

        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ExternallyManagedDeviceSlot for VllmConnectorSlot {
    fn advance_computed_position(&mut self, num_tokens: usize) -> Result<(), SlotError> {
        if self.current_position + num_tokens > self.sequence().total_tokens() {
            return Err(SlotError::InvalidOperation(format!(
                "cannot advance computed position from {} by {num_tokens} tokens, total tokens is {}",
                self.current_position,
                self.sequence().total_tokens()
            )));
        }

        tracing::debug!(
            "advancing computed position by {} tokens from {} to {}",
            num_tokens,
            self.current_position,
            self.current_position + num_tokens
        );

        self.current_position += num_tokens;
        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(request_id = self.request_id))]
    fn append_mutable_device_blocks(&mut self, block_ids: &[BlockId]) -> Result<(), SlotError> {
        let count = block_ids.len();
        self.device_blocks.extend(block_ids);
        tracing::debug!(
            "appended {} mutable device blocks to slot; total device blocks: {}",
            count,
            self.num_device_blocks_allocated()
        );

        Ok(())
    }
}

impl VllmConnectorSlot {
    /// this method does two things which are related:
    /// 1. creates transfer engine offload request
    /// 2. creates matching connector worker transfer request
    ///
    /// these requests share the same uuid.
    ///
    /// the worker request triggers the transfer when sufficient forward pass progress has been made.
    fn offload_blocks(
        &mut self,
        block_ids: &[BlockId],
        token_blocks: &[TokenBlock],
        priorities: &[u32],
    ) -> Result<(), SlotError> {
        // Check if slot is in Finishing state before creating operations
        // If we're finishing, don't create new operations
        if matches!(self.state, SlotState::Finishing | SlotState::Finished) {
            return Ok(());
        }

        assert!(block_ids.len() == token_blocks.len());
        assert!(block_ids.len() == priorities.len());
        let operation_id = uuid::Uuid::new_v4();

        let xfer_req = LocalTransferRequest::Offload(LocalOffloadRequest::new(
            self.request_id.clone(),
            block_ids.to_vec(),
            token_blocks.to_vec(),
            priorities.to_vec(),
            operation_id,
        ));

        // ONE metadata operation PER GROUP LEG, each with its deterministic
        // per-leg uuid — the transfer engine stamps the same uuids on the
        // wire legs, and the worker-side rendezvous (one Worker + one
        // Transfer per `(request_id, uuid)`) requires the 1:1 match.
        // Group 0's leg uuid IS the logical op uuid, so single-group
        // deployments record exactly the same op as before.
        for group in 0..self.num_groups {
            let worker_req = WorkerTransferRequest {
                request_id: self.request_id.clone(),
                uuid: derive_leg_uuid(operation_id, group),
                transfer_type: TransferType::Store,
                request_type: RequestType::Scheduled,
            };
            self.append_pending_operation(worker_req);
        }

        if let Err(e) = self.xfer_tx.send(xfer_req) {
            tracing::error!("Failed to send transfer request: {:?}", e);
            return Err(SlotError::InvalidOperation(format!(
                "Transfer engine unavailable: {}; aborting offload",
                e
            )));
        }

        tracing::debug!(
            request_id = self.request_id,
            operation_id = %operation_id,
            legs = self.num_groups,
            "offloading {} blocks to host",
            block_ids.len()
        );

        Ok(())
    }

    fn onboard_blocks(
        &mut self,
        src_blocks: Box<dyn AnyBlocks>,
        dst_block_ids: Vec<BlockId>,
    ) -> Result<(), SlotError> {
        debug_assert_eq!(src_blocks.len(), dst_block_ids.len());

        let num_blocks = src_blocks.len();
        let src_storage_pool = src_blocks.storage_pool();
        let operation_id = uuid::Uuid::new_v4();

        let xfer_req = LocalTransferRequest::Onboard(LocalOnboardRequest::new(
            self.request_id.clone(),
            src_blocks,
            dst_block_ids,
            operation_id,
        ));

        // ONE metadata operation PER GROUP LEG, each with its deterministic
        // per-leg uuid — the transfer engine stamps the same uuids on the
        // wire legs; the leader's metadata counts Immediate ops per request
        // (expected_immediate_ops) and the worker's slot completion counts
        // ops per request, so an onboard finishes only when ALL legs did.
        for group in 0..self.num_groups {
            let worker_req = WorkerTransferRequest {
                request_id: self.request_id.clone(),
                uuid: derive_leg_uuid(operation_id, group),
                transfer_type: TransferType::Load,
                request_type: RequestType::Immediate,
            };
            self.append_pending_operation(worker_req);
        }

        if let Err(e) = self.xfer_tx.send(xfer_req) {
            tracing::error!("Failed to send transfer request: {:?}", e);
            return Err(SlotError::InvalidOperation(format!(
                "Transfer engine unavailable: {}; aborting offload",
                e
            )));
        }

        tracing::debug!(
            request_id = self.request_id,
            operation_id = %operation_id,
            legs = self.num_groups,
            "start onboarding {} blocks from {:?} to device",
            num_blocks,
            src_storage_pool,
        );

        Ok(())
    }

    fn append_pending_operation(&mut self, operation: WorkerTransferRequest) {
        if let Some(pending_operations) = self.pending_operations.as_mut() {
            pending_operations.push(operation);
        } else {
            self.pending_operations = Some(vec![operation]);
        }
    }
}

enum LocalTransferRequest {
    Offload(LocalOffloadRequest),
    Onboard(LocalOnboardRequest),
}

/// Deterministic per-leg uuid for group `group` of the logical op `logical`:
/// the logical uuid with its low 7 bytes XOR-folded with the group index
/// (the version/variant nibbles in bytes 6/8 stay untouched, so the result
/// remains a well-formed v4-shaped uuid).
///
/// Determinism is the load-bearing property: the leader-side slot records
/// one metadata operation PER LEG with `derive_leg_uuid(op, g)`, and the
/// transfer engine stamps the SAME uuid on each group's wire request —
/// without threading state between them. The worker-side connector
/// scheduler rendezvous keys on `(request_id, uuid)` and supports exactly
/// one Worker + one Transfer per uuid, so N legs sharing one uuid would
/// panic ("invalid combination of request sources") or park a controller
/// forever. Group 0 folds to the logical uuid unchanged, keeping
/// single-group deployments byte-identical on the wire.
fn derive_leg_uuid(logical: uuid::Uuid, group: usize) -> uuid::Uuid {
    let mut bytes = *logical.as_bytes();
    let group_le = (group as u64).to_le_bytes();
    for (i, b) in group_le.iter().enumerate().take(7) {
        bytes[9 + i] ^= b;
    }
    uuid::Uuid::from_bytes(bytes)
}

struct LocalOffloadRequest {
    request_id: String,
    block_ids: Vec<BlockId>,
    token_blocks: Vec<TokenBlock>,
    /// Priorities for each block, used to set BasicMetadata.priority during offload.
    priorities: Vec<u32>,
    operation_id: uuid::Uuid,
}

impl LocalOffloadRequest {
    pub fn new(
        request_id: String,
        block_ids: Vec<BlockId>,
        token_blocks: Vec<TokenBlock>,
        priorities: Vec<u32>,
        operation_id: uuid::Uuid,
    ) -> Self {
        debug_assert!(block_ids.len() == token_blocks.len());
        debug_assert!(block_ids.len() == priorities.len());
        Self {
            request_id,
            block_ids,
            token_blocks,
            priorities,
            operation_id,
        }
    }
}

struct LocalOnboardRequest {
    request_id: String,
    src_blocks: Box<dyn AnyBlocks>,
    dst_block_ids: Vec<BlockId>,
    operation_id: uuid::Uuid,
}

impl LocalOnboardRequest {
    pub fn new(
        request_id: String,
        src_blocks: Box<dyn AnyBlocks>,
        dst_block_ids: Vec<BlockId>,
        operation_id: uuid::Uuid,
    ) -> Self {
        debug_assert!(src_blocks.len() == dst_block_ids.len());
        Self {
            request_id,
            src_blocks,
            dst_block_ids,
            operation_id,
        }
    }
}

struct LocalTransferEngine {
    block_manager: VllmBlockManager,
    leader: Arc<KvbmLeader>,
    /// HMA group count captured from the leader at engine construction —
    /// the SAME capture the slot manager stamps into every slot, so the
    /// metadata ops (one per leg) and the wire legs always agree. Must be
    /// constructed after the worker handshake (all callers await
    /// `wait_worker_sync_ready` first).
    num_groups: usize,
    xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
}

impl LocalTransferEngine {
    pub fn new(
        block_manager: VllmBlockManager,
        leader: Arc<KvbmLeader>,
        num_groups: usize,
        xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
    ) -> Self {
        Self {
            block_manager,
            leader,
            num_groups,
            xfer_rx,
        }
    }

    // build an adapted TaskTracker:
    // https://docs.rs/tokio-util/latest/tokio_util/task/task_tracker/struct.TaskTracker.html
    //
    // this should track completions via atomic counters using the dynamo prometheus metrics
    // - critical_tasks: labels - success, failure, cancelled
    //
    // should spawn any task/future that returns either any task that can be converted to a
    // Result<CompletionStatus, String> where CompletionStatus is an enum with Ok and Cancelled.
    // anyhow::Result<()> can be considered non-cancellable and coerced to Ok(CompletionStatus::Ok)
    // tasks allowed to cancel should return a CompletionStatus.
    //
    // This should be a composable unit that we can layer on specialized types of critical tasks
    // with their own sets of custom metrics.
    async fn execute(
        &mut self,
        cancellation_token: CancellationToken,
        task_handle: Handle,
        task_token: CancellationToken,
        kvbm_metrics: KvbmMetrics,
    ) -> anyhow::Result<()> {
        let (onboard_tx, mut onboard_rx) = mpsc::unbounded_channel::<LocalOnboardRequest>();
        let (offload_tx, mut offload_rx) = mpsc::unbounded_channel::<LocalOffloadRequest>();

        // Clone resources needed for tasks
        let block_manager_offload = self.block_manager.clone();
        let leader_offload = Arc::clone(&self.leader);
        let leader_onboard = Arc::clone(&self.leader);
        let num_groups_onboard = self.num_groups;
        let num_groups_offload = self.num_groups;

        let kvbm_metrics_onboard = kvbm_metrics.clone();
        let kvbm_metrics_offload = kvbm_metrics.clone();

        let onboard_task = CriticalTaskExecutionHandle::new_with_runtime(
            move |cancellation_token_onboard| async move {
                while let Some(req) = onboard_rx.recv().await {
                    if cancellation_token_onboard.is_cancelled() {
                        tracing::debug!("LocalOnboardTask: received cancellation signal");
                        break;
                    }
                    if let Err(e) = process_onboard_request(
                        req,
                        &leader_onboard,
                        num_groups_onboard,
                        kvbm_metrics_onboard.clone(),
                    )
                    .await
                    {
                        tracing::error!("LocalOnboardTask: error processing request: {:?}", e);
                    }
                }
                Ok(())
            },
            task_token.clone(),
            "LocalOnboardTask",
            &task_handle,
        )
        .unwrap();
        let offload_task = CriticalTaskExecutionHandle::new_with_runtime(
            move |cancellation_token_offload| async move {
                while let Some(req) = offload_rx.recv().await {
                    if cancellation_token_offload.is_cancelled() {
                        tracing::debug!("LocalOffloadTask: received cancellation signal");
                        break;
                    }

                    let request_id = req.request_id.clone();
                    let operation_id = req.operation_id;

                    if let Err(e) = process_offload_request(
                        req,
                        &block_manager_offload,
                        &leader_offload,
                        num_groups_offload,
                        kvbm_metrics_offload.clone(),
                    )
                    .await
                    {
                        tracing::error!("LocalOffloadTask: error processing request: {:?}", e);

                        // Create fake/immediate transfer requests that complete
                        // instantly — ONE PER GROUP LEG. Otherwise, the worker
                        // side might get stuck and leak memory: the slot already
                        // recorded N metadata ops for this request, and each
                        // parked rendezvous must be unblocked with its own leg
                        // uuid (group 0's leg uuid IS the logical op uuid).
                        for group in 0..num_groups_offload {
                            let fake_xfer = BlockTransferRequest {
                                from_pool: BlockTransferPool::Device, // Use valid Device->Host transfer type
                                to_pool: BlockTransferPool::Host, // (offload path, but no blocks)
                                blocks: vec![],                   // Empty - nothing to transfer
                                group,
                                connector_req: Some(LeaderTransferRequest {
                                    request_id: request_id.clone(),
                                    uuid: derive_leg_uuid(operation_id, group),
                                    requirement: None,
                                    request_type: RequestType::Immediate, // Immediate = completes instantly
                                }),
                            };

                            match leader_offload.transfer_blocks_request(fake_xfer).await {
                                Ok(notify_receiver) => {
                                    // Wait for the fake transfer to "complete" (should be instant)
                                    let _ = notify_receiver.await;
                                }
                                Err(_xfer_err) => {
                                    // Failed to create completion notification - error already logged above
                                }
                            }
                        }
                    }
                }
                Ok(())
            },
            task_token,
            "LocalOffloadTask",
            &task_handle,
        )
        .unwrap();

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    tracing::debug!("LocalTransferEngine: received cancellation signal");
                    break;
                }
                req = self.xfer_rx.recv() => {
                    match req {
                        Some(req) => {
                            match req {
                                LocalTransferRequest::Offload(offload_req) => {
                                    if let Err(e) = offload_tx.send(offload_req) {
                                        tracing::error!("LocalTransferEngine: error sending offload request: {:?}", e);
                                    }
                                }
                                LocalTransferRequest::Onboard(onboard_req) => {
                                    if let Err(e) = onboard_tx.send(onboard_req) {
                                        tracing::error!("LocalTransferEngine: error sending onboard request: {:?}", e);
                                    }
                                }
                            }
                        }
                        None => {
                            tracing::debug!("LocalTransferEngine: channel closed");
                            break;
                        }
                    }
                }
            }
        }

        tracing::debug!("LocalTransferEngine: shutting down");

        // drop all tx channels
        drop(onboard_tx);
        drop(offload_tx);

        onboard_task.cancel();
        offload_task.cancel();

        if let Err(e) = onboard_task.join().await {
            tracing::error!("LocalOnboardTask failed: {:?}", e);
        }
        if let Err(e) = offload_task.join().await {
            tracing::error!("LocalOffloadTask failed: {:?}", e);
        }

        tracing::debug!("LocalTransferEngine: shutdown complete");
        Ok(())
    }
}

async fn process_offload_request(
    offload_req: LocalOffloadRequest,
    block_manager: &VllmBlockManager,
    leader: &Arc<KvbmLeader>,
    num_groups: usize,
    kvbm_metrics: KvbmMetrics,
) -> anyhow::Result<()> {
    let request_id = offload_req.request_id.clone();
    let operation_id = offload_req.operation_id;

    tracing::debug!(
        "Processing offload request for {} blocks",
        offload_req.block_ids.len()
    );

    // Determine if we should bypass CPU memory (G2) and offload directly from GPU (G1) to Disk (G3)
    let bypass_cpu_mem = should_bypass_cpu_cache();

    if bypass_cpu_mem {
        // Direct G1 -> G3 path (Device to Disk, bypassing Host)
        tracing::debug!(
            request_id = %request_id,
            operation_id = %operation_id,
            "offloading directly to disk (bypassing host)"
        );

        process_offload_to_storage(
            offload_req,
            block_manager.disk().unwrap(),
            BlockTransferPool::Disk,
            leader,
            num_groups,
            kvbm_metrics,
            &request_id,
            &operation_id,
            "disk",
        )
        .await?;
    } else {
        // Standard path: G1 -> G2 (Device to Host)
        process_offload_to_storage(
            offload_req,
            block_manager.host().unwrap(),
            BlockTransferPool::Host,
            leader,
            num_groups,
            kvbm_metrics,
            &request_id,
            &operation_id,
            "host",
        )
        .await?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn process_offload_to_storage<S, L, M>(
    offload_req: LocalOffloadRequest,
    storage_pool: &dyn BlockPool<S, L, M>,
    transfer_pool: BlockTransferPool,
    leader: &Arc<KvbmLeader>,
    num_groups: usize,
    kvbm_metrics: KvbmMetrics,
    request_id: &str,
    operation_id: &uuid::Uuid,
    storage_name: &str,
) -> anyhow::Result<()>
where
    S: Storage + NixlRegisterableStorage,
    L: LocalityProvider,
    M: BlockMetadata,
{
    // 1. Acquire mutable blocks
    let blocks = storage_pool
        .allocate_blocks(offload_req.block_ids.len())
        .await?;
    let token_blocks = offload_req.token_blocks;

    let allocated_block_ids: Vec<usize> = blocks.iter().map(|b| b.block_id()).collect();
    let block_pairs: Vec<(usize, usize)> = offload_req
        .block_ids
        .into_iter()
        .zip(allocated_block_ids)
        .collect();
    let num_blocks = block_pairs.len();

    tracing::debug!(
        request_id = request_id,
        operation_id = %operation_id,
        "offload to {} - stage 1 complete",
        storage_name
    );

    // 2. Apply token blocks and set priorities
    let mut blocks_to_register = Vec::new();
    let priorities = offload_req.priorities;

    for ((mut mutable_block, token_block), priority) in
        blocks.into_iter().zip(token_blocks).zip(priorities)
    {
        mutable_block
            .apply_token_block(token_block.clone())
            .map_err(|e| anyhow::anyhow!("failed to apply token block: {:?}", e))?;

        // Set the priority on the block's metadata so it flows through to downstream processing
        let updated_metadata = mutable_block.metadata().with_priority(priority);
        mutable_block.update_metadata(updated_metadata);

        blocks_to_register.push(mutable_block);
    }
    tracing::debug!(
        request_id = request_id,
        operation_id = %operation_id,
        "offload to {} - stage 2 complete",
        storage_name
    );

    // 3. Issue the offload: ONE logical op fanned out to one request PER
    // GROUP (same block pairs — the tier's block ids index every group's
    // worker-side pool; each worker holds one pool per group), each leg with
    // its own deterministic uuid. All legs are sent before any is awaited.
    // Single-group (num_groups == 1) sends the one request exactly as
    // before, with the logical op's uuid (derive_leg_uuid identity).
    let notify_receivers = fan_out_group_legs(
        leader,
        num_groups,
        &offload_req.request_id,
        offload_req.operation_id,
        BlockTransferPool::Device,
        transfer_pool,
        block_pairs,
        RequestType::Scheduled,
        "offload",
    )
    .await;
    tracing::debug!(
        request_id = request_id,
        operation_id = %operation_id,
        "offload to {} - stage 3 complete ({} group legs)",
        storage_name,
        num_groups
    );
    let notify_receivers = notify_receivers?;

    // 4. Wait for ALL group legs to complete. The metrics count the
    // completed LOGICAL op once (not once per group leg).
    await_all_legs(notify_receivers, "offload").await?;
    match transfer_pool {
        BlockTransferPool::Disk => {
            kvbm_metrics.offload_blocks_d2d.inc_by(num_blocks as u64);
        }
        BlockTransferPool::Host => {
            kvbm_metrics.offload_blocks_d2h.inc_by(num_blocks as u64);
        }
        _ => {}
    }
    tracing::debug!(
        "Offloading transfer to {} completed successfully",
        storage_name
    );
    tracing::debug!(
        request_id = request_id,
        operation_id = %operation_id,
        "offload to {} - stage 4 complete",
        storage_name
    );

    // 5. Register the mutable blocks
    let immutable_blocks = storage_pool.register_blocks(blocks_to_register).await?;

    tracing::debug!(
        request_id = request_id,
        operation_id = %operation_id,
        "registered {} blocks to {}",
        immutable_blocks.len(),
        storage_name
    );

    Ok(())
}

/// Send one wire request PER GROUP LEG for a logical offload/onboard op.
///
/// Each leg carries its own deterministic uuid (`derive_leg_uuid`) and its
/// group tag — the worker-side connector scheduler rendezvous keys on
/// `(request_id, uuid)` and supports exactly one Worker + one Transfer per
/// uuid, so legs must not share the logical op's uuid. All legs are sent
/// before any is awaited; if a leg's send fails after earlier legs were
/// already sent, the in-flight legs are logged as orphaned rather than
/// silently `?`-returned past.
#[allow(clippy::too_many_arguments)]
async fn fan_out_group_legs(
    leader: &KvbmLeader,
    num_groups: usize,
    request_id: &str,
    operation_id: uuid::Uuid,
    from_pool: BlockTransferPool,
    to_pool: BlockTransferPool,
    block_pairs: Vec<(usize, usize)>,
    request_type: RequestType,
    op_name: &str,
) -> anyhow::Result<Vec<tokio::sync::oneshot::Receiver<()>>> {
    let mut notify_receivers = Vec::with_capacity(num_groups);
    for group in 0..num_groups {
        let block_xfer_req = BlockTransferRequest {
            from_pool,
            to_pool,
            blocks: block_pairs.clone(),
            group,
            connector_req: Some(LeaderTransferRequest {
                request_id: request_id.to_string(),
                uuid: derive_leg_uuid(operation_id, group),
                requirement: None,
                request_type,
            }),
        };
        match leader.transfer_blocks_request(block_xfer_req).await {
            Ok(receiver) => notify_receivers.push(receiver),
            Err(e) => {
                if !notify_receivers.is_empty() {
                    tracing::error!(
                        request_id = %request_id,
                        operation_id = %operation_id,
                        op_name,
                        in_flight_legs = notify_receivers.len(),
                        failed_group = group,
                        "{op_name}: leg send failed after earlier legs were already in flight; \
                         orphaning them (their acks will be dropped)"
                    );
                }
                return Err(e);
            }
        }
    }
    Ok(notify_receivers)
}

/// Await ALL group legs of one logical op, collecting (not short-circuiting
/// on) failures so every in-flight leg is drained. The caller counts the
/// logical op once, only when all legs completed.
async fn await_all_legs(
    notify_receivers: Vec<tokio::sync::oneshot::Receiver<()>>,
    op_name: &str,
) -> anyhow::Result<()> {
    let mut failed_legs = Vec::new();
    for (group, notify_receiver) in notify_receivers.into_iter().enumerate() {
        if notify_receiver.await.is_err() {
            failed_legs.push(group);
        }
    }
    if failed_legs.is_empty() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "{op_name} transfer completion notification failed for group legs {failed_legs:?}"
        ))
    }
}

async fn process_onboard_request(
    onboard_req: LocalOnboardRequest,
    leader: &Arc<KvbmLeader>,
    num_groups: usize,
    kvbm_metrics: KvbmMetrics,
) -> anyhow::Result<()> {
    let request_id = &onboard_req.request_id;
    let operation_id = &onboard_req.operation_id;

    // extract source block ids
    let src_block_ids = onboard_req.src_blocks.block_ids();

    // create block pairs
    let block_pairs = src_block_ids
        .iter()
        .zip(onboard_req.dst_block_ids.iter())
        .map(|(src, dst)| (*src, *dst))
        .collect::<Vec<_>>();

    // Issue the onboard: ONE logical op fanned out to one request PER GROUP
    // (same block pairs; each group's worker-side pools), each leg with its
    // own deterministic uuid. All legs are sent before any is awaited.
    let notify_receivers = fan_out_group_legs(
        leader,
        num_groups,
        request_id,
        *operation_id,
        onboard_req.src_blocks.storage_pool(),
        BlockTransferPool::Device,
        block_pairs,
        RequestType::Immediate,
        "onboard",
    )
    .await?;

    // Wait for ALL group legs; count the completed LOGICAL op once (not
    // once per group leg).
    await_all_legs(notify_receivers, "onboard").await?;
    if onboard_req.src_blocks.storage_pool() == BlockTransferPool::Host {
        kvbm_metrics
            .onboard_blocks_h2d
            .inc_by(onboard_req.src_blocks.len() as u64);
    } else if onboard_req.src_blocks.storage_pool() == BlockTransferPool::Disk {
        kvbm_metrics
            .onboard_blocks_d2d
            .inc_by(onboard_req.src_blocks.len() as u64);
    }
    tracing::debug!("Onboarding transfer completed successfully");

    Ok(())
}

// todo move to core lib
pub trait AnyBlocks: Send {
    fn len(&self) -> usize;
    fn storage_pool(&self) -> BlockTransferPool;
    fn block_ids(&self) -> Vec<BlockId>;
}

struct AnyImmutableBlocks<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    blocks: Vec<ImmutableBlock<S, L, M>>,
    storage_pool: BlockTransferPool,
}

impl<L: LocalityProvider, M: BlockMetadata> AnyImmutableBlocks<DevbarStorage, L, M> {
    pub fn new(blocks: Vec<ImmutableBlock<DevbarStorage, L, M>>) -> Self {
        Self {
            blocks,
            storage_pool: BlockTransferPool::Host,
        }
    }
}

impl<L: LocalityProvider, M: BlockMetadata> AnyImmutableBlocks<DiskStorage, L, M> {
    pub fn new(blocks: Vec<ImmutableBlock<DiskStorage, L, M>>) -> Self {
        Self {
            blocks,
            storage_pool: BlockTransferPool::Disk,
        }
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> AnyImmutableBlocks<S, L, M> {
    pub fn storage_pool(&self) -> BlockTransferPool {
        self.storage_pool
    }

    pub fn block_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(|b| b.block_id()).collect()
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> AnyBlocks for AnyImmutableBlocks<S, L, M> {
    fn len(&self) -> usize {
        self.len()
    }

    fn storage_pool(&self) -> BlockTransferPool {
        self.storage_pool()
    }

    fn block_ids(&self) -> Vec<BlockId> {
        self.block_ids()
    }
}

#[cfg(test)]
mod connector_tests {
    use super::*;
    use crate::block_manager::cache_stats::CacheStatsTracker;
    use dynamo_llm::tokens::{SaltHash, SequenceHash, Tokens};
    use std::sync::Arc;
    use tokio::sync::mpsc;

    const BLOCK_SIZE: usize = 32;
    const SALT_HASH: SaltHash = 12345;

    /// Creates a test slot with `num_tokens` tokens and the given priority threshold.
    /// Returns the slot and the receiving end of the transfer channel for inspecting offload requests.
    fn create_test_slot(
        num_tokens: usize,
        offload_min_priority: u32,
    ) -> (
        VllmConnectorSlot,
        mpsc::UnboundedReceiver<LocalTransferRequest>,
    ) {
        create_test_slot_with_groups(num_tokens, offload_min_priority, 1)
    }

    /// Variant with an explicit HMA group count (the slot's per-op leg
    /// count for offload/onboard metadata).
    fn create_test_slot_with_groups(
        num_tokens: usize,
        offload_min_priority: u32,
        num_groups: usize,
    ) -> (
        VllmConnectorSlot,
        mpsc::UnboundedReceiver<LocalTransferRequest>,
    ) {
        let tokens: Vec<u32> = (1..=num_tokens as u32).collect();
        let (xfer_tx, xfer_rx) = mpsc::unbounded_channel();
        let cache_stats = Arc::new(CacheStatsTracker::new(None));
        let slot = VllmConnectorSlot::new_for_test(
            "test-req".to_string(),
            Tokens::from(tokens),
            SALT_HASH,
            BLOCK_SIZE,
            xfer_tx,
            cache_stats,
            offload_min_priority,
            num_groups,
        );
        (slot, xfer_rx)
    }

    /// Generates block IDs starting from `start`.
    fn block_ids(start: usize, count: usize) -> Vec<usize> {
        (start..start + count).collect()
    }

    fn external_hashes(start: SequenceHash, count: usize) -> Vec<SequenceHash> {
        (0..count)
            .map(|offset| start + offset as SequenceHash)
            .collect()
    }

    /// Drains all pending offload requests from the channel and returns their block IDs.
    fn drain_offload_block_ids(
        rx: &mut mpsc::UnboundedReceiver<LocalTransferRequest>,
    ) -> Vec<Vec<usize>> {
        let mut results = Vec::new();
        while let Ok(req) = rx.try_recv() {
            if let LocalTransferRequest::Offload(offload) = req {
                results.push(offload.block_ids);
            }
        }
        results
    }

    // ---------------------------------------------------------------
    // Test 1: vLLM pattern — append_mutable then apply with empty blocks
    // ---------------------------------------------------------------
    #[test]
    fn test_vllm_pattern_no_double_add() {
        let num_tokens = 96; // 3 blocks of 32
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);
        let blocks = block_ids(100, 3);

        // Step 1: append_mutable (from update_state_after_alloc)
        slot.append_mutable_device_blocks(&blocks).unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 3);

        // Step 2: apply_scheduler_output with empty blocks (vLLM pattern)
        slot.apply_scheduler_output(&[], &[], 0, num_tokens, None, None)
            .unwrap();

        // device_blocks should still be exactly 3 — no double-add
        assert_eq!(slot.num_device_blocks_allocated(), 3);
    }

    // ---------------------------------------------------------------
    // Test 2: TRT-LLM pattern — append_mutable then apply with SAME blocks
    //         The dedup guard must prevent device_blocks from doubling.
    // ---------------------------------------------------------------
    #[test]
    fn test_trtllm_pattern_no_double_add() {
        let num_tokens = 96; // 3 blocks of 32
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);
        let blocks = block_ids(100, 3);

        // Step 1: append_mutable (from update_state_after_alloc)
        slot.append_mutable_device_blocks(&blocks).unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 3);

        // Step 2: apply_scheduler_output with THE SAME blocks (TRT-LLM pattern)
        // Without the dedup guard, this doubles device_blocks to len=6.
        slot.apply_scheduler_output(&[], &blocks, 0, num_tokens, None, None)
            .unwrap();

        // device_blocks must still be exactly 3 — dedup guard prevented the double-add
        assert_eq!(slot.num_device_blocks_allocated(), 3);
    }

    #[test]
    fn test_trtllm_external_hashes_are_assigned_to_completed_blocks() {
        let num_tokens = 96; // 3 blocks of 32
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);
        let blocks = block_ids(100, 3);
        let external_hashes = external_hashes(10_000, 3);

        slot.append_mutable_device_blocks(&blocks).unwrap();
        slot.apply_scheduler_output(&[], &blocks, 0, num_tokens, None, Some(&external_hashes))
            .unwrap();

        let completed_blocks = slot.sequence.blocks();
        assert_eq!(completed_blocks.len(), 3);
        assert_eq!(
            completed_blocks[0].external_sequence_hash(),
            Some(external_hashes[0])
        );
        assert_eq!(completed_blocks[0].external_parent_sequence_hash(), None);
        assert_eq!(
            completed_blocks[1].external_sequence_hash(),
            Some(external_hashes[1])
        );
        assert_eq!(
            completed_blocks[1].external_parent_sequence_hash(),
            Some(external_hashes[0])
        );
        assert_eq!(
            completed_blocks[2].external_sequence_hash(),
            Some(external_hashes[2])
        );
        assert_eq!(
            completed_blocks[2].external_parent_sequence_hash(),
            Some(external_hashes[1])
        );
    }

    #[test]
    #[should_panic(expected = "external_sequence_hash mismatch")]
    fn test_trtllm_external_hash_chain_mismatch_panics() {
        let num_tokens = 96; // 3 blocks of 32
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);
        let blocks = block_ids(100, 3);
        let external_hashes = external_hashes(20_000, 3);

        slot.append_mutable_device_blocks(&blocks).unwrap();
        slot.apply_scheduler_output(&[], &blocks, 0, num_tokens, None, Some(&external_hashes))
            .unwrap();

        let mismatched_hashes = vec![
            external_hashes[0],
            external_hashes[1] + 1,
            external_hashes[2],
        ];
        slot.apply_scheduler_output(&[], &blocks, 0, num_tokens, None, Some(&mismatched_hashes))
            .unwrap();
    }

    // ---------------------------------------------------------------
    // Test 3: Decode adds a new block correctly
    // ---------------------------------------------------------------
    #[test]
    fn test_decode_block_added_correctly() {
        let num_tokens = 96; // 3 blocks
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);
        let prefill_blocks = block_ids(100, 3);

        // Prefill: append + apply with empty blocks (vLLM pattern)
        slot.append_mutable_device_blocks(&prefill_blocks).unwrap();
        slot.apply_scheduler_output(&[], &[], 0, num_tokens, None, None)
            .unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 3);

        // Decode: new block at boundary (token 96 = block 3)
        let decode_block = block_ids(200, 1);
        let decode_token: Vec<u32> = vec![9999];
        slot.apply_scheduler_output(&decode_token, &decode_block, 95, 1, None, None)
            .unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 4);
    }

    // ---------------------------------------------------------------
    // Test 4: Multiple append_mutable calls accumulate
    // ---------------------------------------------------------------
    #[test]
    fn test_append_mutable_is_additive() {
        let (mut slot, _rx) = create_test_slot(128, 0);

        slot.append_mutable_device_blocks(&block_ids(100, 2))
            .unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 2);

        slot.append_mutable_device_blocks(&block_ids(200, 1))
            .unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 3);
    }

    // ---------------------------------------------------------------
    // Test 5: Offload sends correct block IDs through channel
    // ---------------------------------------------------------------
    #[test]
    fn test_offload_sends_correct_block_ids() {
        let num_tokens = 96; // 3 blocks
        let (mut slot, mut rx) = create_test_slot(num_tokens, 0);
        let blocks = block_ids(100, 3);

        // Prefill: append blocks, then apply with num_scheduled_tokens=96.
        // All 3 blocks (96/32) become offload candidates since evaluated_blocks starts at 0.
        // Empty tokens → Prefilling state, and next_position(96) == total_tokens(96)
        // so the early-return does not fire and offload proceeds.
        slot.append_mutable_device_blocks(&blocks).unwrap();
        slot.apply_scheduler_output(&[], &[], 0, num_tokens, None, None)
            .unwrap();

        let offloads = drain_offload_block_ids(&mut rx);
        assert_eq!(offloads.len(), 1, "expected exactly one offload batch");
        assert_eq!(offloads[0], vec![100, 101, 102]);
    }

    // ---------------------------------------------------------------
    // Test 6: Priority filtering offloads only blocks above threshold
    // ---------------------------------------------------------------
    #[test]
    fn test_priority_filtering_offloads_correct_count() {
        let num_tokens = 96; // 3 blocks
        let (mut slot, mut rx) = create_test_slot(num_tokens, 30);
        let blocks = block_ids(100, 3);
        let priorities: Vec<u32> = vec![80, 80, 10]; // 2 above threshold, 1 below

        // Use the TRT-LLM pattern: append_mutable first, then apply with same blocks + priorities.
        // The dedup guard prevents the double-add, but priorities are still processed.
        slot.append_mutable_device_blocks(&blocks).unwrap();
        slot.apply_scheduler_output(&[], &blocks, 0, num_tokens, Some(&priorities), None)
            .unwrap();

        // device_blocks should be 3 (dedup prevented doubling)
        assert_eq!(slot.num_device_blocks_allocated(), 3);

        let offloads = drain_offload_block_ids(&mut rx);
        assert_eq!(offloads.len(), 1);
        // Only blocks with priority >= 30 should be offloaded (first 2)
        assert_eq!(offloads[0], vec![100, 101]);
    }

    // ---------------------------------------------------------------
    // Test 7: Priority filtering terminates further offloading
    // ---------------------------------------------------------------
    #[test]
    fn test_priority_filtering_terminates_offload() {
        let num_tokens = 128; // 4 blocks
        let (mut slot, mut rx) = create_test_slot(num_tokens, 30);
        let blocks = block_ids(100, 4);
        // First 2 high, then 2 low — offload terminates at block 2
        let priorities: Vec<u32> = vec![80, 80, 10, 10];

        slot.append_mutable_device_blocks(&blocks).unwrap();
        slot.apply_scheduler_output(&[], &blocks, 0, num_tokens, Some(&priorities), None)
            .unwrap();

        let offloads = drain_offload_block_ids(&mut rx);
        assert_eq!(offloads[0], vec![100, 101]); // only 2 offloaded

        // Now simulate a decode iteration that crosses a block boundary.
        // Because offload was terminated, no further offloading should happen.
        let decode_block = block_ids(200, 1);
        let decode_token: Vec<u32> = vec![9999];
        slot.apply_scheduler_output(&decode_token, &decode_block, 127, 1, None, None)
            .unwrap();

        let further_offloads = drain_offload_block_ids(&mut rx);
        assert!(
            further_offloads.is_empty(),
            "no offloads should occur after termination"
        );
    }

    // ---------------------------------------------------------------
    // Test 8: evaluated_blocks advances correctly across iterations
    // ---------------------------------------------------------------
    #[test]
    fn test_evaluated_blocks_advances_correctly() {
        // 128 tokens = 4 blocks. We'll process in 2 chunks of 64 tokens.
        let num_tokens = 128;
        let (mut slot, mut rx) = create_test_slot(num_tokens, 0);
        let blocks = block_ids(100, 4);

        slot.append_mutable_device_blocks(&blocks).unwrap();

        // Chunk 1: schedule first 64 tokens → evaluates blocks 0,1
        slot.apply_scheduler_output(&[], &[], 0, 64, None, None)
            .unwrap();
        let offloads_1 = drain_offload_block_ids(&mut rx);
        assert_eq!(offloads_1.len(), 1);
        assert_eq!(offloads_1[0], vec![100, 101]); // blocks 0,1

        // Chunk 2: schedule next 64 tokens → evaluates blocks 2,3
        // (uses cached_request pattern: empty tokens, empty blocks)
        slot.apply_scheduler_output(&[], &[], 64, 64, None, None)
            .unwrap();
        let offloads_2 = drain_offload_block_ids(&mut rx);
        assert_eq!(offloads_2.len(), 1);
        assert_eq!(offloads_2[0], vec![102, 103]); // blocks 2,3

        assert_eq!(slot.num_device_blocks_allocated(), 4);
    }

    // ---------------------------------------------------------------
    // Test 9: Partial overlap dedup — suffix/prefix contract
    //         [10,11,12] + apply([12,13]) → [10,11,12,13]
    // ---------------------------------------------------------------
    #[test]
    fn test_partial_overlap_dedup() {
        let num_tokens = 128; // 4 blocks
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        // Step 1: append_mutable with first 3 blocks
        slot.append_mutable_device_blocks(&[10, 11, 12]).unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 3);

        // Step 2: apply_scheduler_output with overlapping blocks [12, 13].
        // Suffix [12] of device_blocks matches prefix [12] of block_ids.
        // Only block 13 is new and gets appended.
        slot.apply_scheduler_output(&[], &[12, 13], 0, 128, None, None)
            .unwrap();

        assert_eq!(slot.num_device_blocks_allocated(), 4);
        assert_eq!(slot.device_blocks_snapshot(), &[10, 11, 12, 13]);
    }

    // ---------------------------------------------------------------
    // Test 10: Priorities work correctly with partial overlap dedup.
    //          Chunk 1 provides priorities for blocks 10-12, chunk 2
    //          overlaps on block 12 and adds block 13 with low priority.
    // ---------------------------------------------------------------
    #[test]
    fn test_partial_overlap_with_priorities() {
        let num_tokens = 128; // 4 blocks
        let (mut slot, mut rx) = create_test_slot(num_tokens, 30);

        // Chunk 1: 3 blocks, all high priority, schedule 96 tokens
        slot.append_mutable_device_blocks(&[10, 11, 12]).unwrap();
        slot.apply_scheduler_output(&[], &[10, 11, 12], 0, 96, Some(&[80, 80, 80]), None)
            .unwrap();

        let offloads_1 = drain_offload_block_ids(&mut rx);
        assert_eq!(offloads_1.len(), 1);
        assert_eq!(offloads_1[0], vec![10, 11, 12]);
        assert_eq!(slot.num_device_blocks_allocated(), 3);

        // Chunk 2: block 13 is new. append_mutable adds it.
        // apply receives [12, 13] with priorities [80, 10].
        // Dedup: suffix [12] matches prefix [12], overlap=1, extends with [13].
        // But block 13 was already added by append_mutable, so the new_ids
        // validation would fail — unless append_mutable already put it there.
        // Actually: append_mutable added 13, so device_blocks = [10,11,12,13].
        // Then apply receives [12,13]: suffix [12,13] matches prefix [12,13], overlap=2.
        // No new blocks to extend. Priorities {12->80, 13->10} stored in HashMap.
        slot.append_mutable_device_blocks(&[13]).unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 4);

        slot.apply_scheduler_output(&[], &[12, 13], 96, 32, Some(&[80, 10]), None)
            .unwrap();

        // Candidate is block 13 (index 3, evaluated_blocks=3).
        // Priority for 13 = 10 (< threshold 30), so no offload.
        let offloads_2 = drain_offload_block_ids(&mut rx);
        assert!(
            offloads_2.is_empty(),
            "block 13 has priority 10 < threshold 30"
        );
        assert_eq!(slot.device_blocks_snapshot(), &[10, 11, 12, 13]);
    }

    // ---------------------------------------------------------------
    // Test 11: Invalid overlap panics — block present but not at tail
    //          [10,11,12] + apply([11,14]) → panic (contract violation)
    // ---------------------------------------------------------------
    #[test]
    #[should_panic(expected = "contract violation")]
    fn test_invalid_overlap_panics() {
        let num_tokens = 128;
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        slot.append_mutable_device_blocks(&[10, 11, 12]).unwrap();

        // block_ids[0]=11 is found at device_blocks[1], but device_blocks[1..] = [11,12]
        // does NOT match block_ids[..2] = [11,14]. Contract violation.
        slot.apply_scheduler_output(&[], &[11, 14], 0, 128, None, None)
            .unwrap();
    }

    // ---------------------------------------------------------------
    // Test 12: Non-contiguous duplicate panics (Case 3)
    //          [10,11,12,13,14] + apply([13,14,10]) → block 10 already
    //          exists but is not part of the suffix/prefix overlap.
    // ---------------------------------------------------------------
    #[test]
    #[should_panic(expected = "contract violation")]
    fn test_non_contiguous_duplicate_panics() {
        let num_tokens = 192; // 6 blocks
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        slot.append_mutable_device_blocks(&[10, 11, 12, 13, 14])
            .unwrap();

        // Overlap: suffix [13,14] matches prefix [13,14], overlap=2.
        // new_ids = [10]. But 10 ∈ device_blocks → contract violation.
        slot.apply_scheduler_output(&[], &[13, 14, 10], 0, 192, None, None)
            .unwrap();
    }

    // ---------------------------------------------------------------
    // Test 13: Over-provision — caller provides all blocks including
    //          those already registered. Full prefix overlap.
    //          [10,11,12] + apply([10,11,12,13,14]) → [10,11,12,13,14]
    // ---------------------------------------------------------------
    #[test]
    fn test_over_provision_dedup() {
        let num_tokens = 160; // 5 blocks
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        slot.append_mutable_device_blocks(&[10, 11, 12]).unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 3);

        // block_ids[0]=10 found at device_blocks[0], suffix_len=3.
        // device_blocks[0..3]=[10,11,12] == block_ids[0..3]=[10,11,12] → overlap=3.
        // new_ids=[13,14], both genuinely new → extend.
        slot.apply_scheduler_output(&[], &[10, 11, 12, 13, 14], 0, 160, None, None)
            .unwrap();

        assert_eq!(slot.num_device_blocks_allocated(), 5);
        assert_eq!(slot.device_blocks_snapshot(), &[10, 11, 12, 13, 14]);
    }

    // ---------------------------------------------------------------
    // Test 14: Chunked re-provision — all blocks re-sent in chunk 2
    //          with additional new blocks appended.
    //          [10,11,12,13,14] + apply([10,11,12,13,14,15,16])
    //          → [10,11,12,13,14,15,16]
    // ---------------------------------------------------------------
    #[test]
    fn test_chunked_reprovision_dedup() {
        let num_tokens = 224; // 7 blocks
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        slot.append_mutable_device_blocks(&[10, 11, 12, 13, 14])
            .unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 5);

        // Full overlap of all 5 existing blocks, plus 2 new.
        slot.apply_scheduler_output(&[], &[10, 11, 12, 13, 14, 15, 16], 0, 224, None, None)
            .unwrap();

        assert_eq!(slot.num_device_blocks_allocated(), 7);
        assert_eq!(slot.device_blocks_snapshot(), &[10, 11, 12, 13, 14, 15, 16]);
    }

    // ---------------------------------------------------------------
    // Multi-group: per-leg uuids + per-leg metadata ops
    // ---------------------------------------------------------------

    #[test]
    fn derive_leg_uuid_is_deterministic_leg0_identity_and_injective() {
        let op = uuid::Uuid::new_v4();
        let other_op = uuid::Uuid::new_v4();

        // Deterministic: same inputs, same leg uuid.
        assert_eq!(derive_leg_uuid(op, 3), derive_leg_uuid(op, 3));

        // Leg 0 folds to the logical uuid — single-group deployments keep
        // the exact same uuids on the wire as before the fan-out.
        assert_eq!(derive_leg_uuid(op, 0), op);

        // Distinct legs of one op, and same-leg uuids of distinct ops.
        let legs: Vec<_> = (0..8).map(|g| derive_leg_uuid(op, g)).collect();
        for g in 0..8 {
            for h in 0..8 {
                if g != h {
                    assert_ne!(legs[g], legs[h], "legs {g} and {h} collided");
                }
                assert_ne!(
                    derive_leg_uuid(other_op, g),
                    derive_leg_uuid(op, g),
                    "different ops collided on leg {g}"
                );
            }
        }

        // Version/variant nibbles (bytes 6/8) survive the fold: the derived
        // uuid stays well-formed v4 / RFC4122.
        for leg in &legs {
            assert_eq!(leg.get_version_num(), op.get_version_num());
            assert_eq!(leg.get_variant(), op.get_variant());
        }
    }

    #[test]
    fn offload_records_one_metadata_operation_per_group() {
        let num_tokens = 96; // 3 blocks
        let (mut slot, mut rx) = create_test_slot_with_groups(num_tokens, 0, 3);

        slot.append_mutable_device_blocks(&block_ids(100, 3))
            .unwrap();
        slot.apply_scheduler_output(&[], &[], 0, num_tokens, None, None)
            .unwrap();

        // ONE wire request goes to the transfer engine (the engine fans it
        // out per group); the SLOT records one metadata op per leg.
        let offloads = drain_offload_block_ids(&mut rx);
        assert_eq!(offloads.len(), 1, "expected exactly one offload batch");

        let ops = slot.take_pending_operations().expect("pending ops");
        assert_eq!(ops.len(), 3, "one metadata op per group leg");
        for (group, op) in ops.iter().enumerate() {
            assert_eq!(op.request_id, "test-req");
            assert_eq!(op.transfer_type, TransferType::Store);
            assert_eq!(op.request_type, RequestType::Scheduled);
            assert_eq!(
                op.uuid,
                derive_leg_uuid(ops[0].uuid, group),
                "leg {group} uuid must be the deterministic derivative of leg 0"
            );
        }
        // All legs distinct, leg 0 is the logical op uuid.
        assert_ne!(ops[0].uuid, ops[1].uuid);
        assert_ne!(ops[0].uuid, ops[2].uuid);
        assert_eq!(ops[0].uuid, derive_leg_uuid(ops[0].uuid, 0));
    }

    /// Minimal [`AnyBlocks`] stand-in so onboard metadata recording can be
    /// tested without real tier blocks.
    struct TestAnyBlocks {
        ids: Vec<BlockId>,
        pool: BlockTransferPool,
    }

    impl AnyBlocks for TestAnyBlocks {
        fn len(&self) -> usize {
            self.ids.len()
        }

        fn storage_pool(&self) -> BlockTransferPool {
            self.pool
        }

        fn block_ids(&self) -> Vec<BlockId> {
            self.ids.clone()
        }
    }

    #[test]
    fn onboard_records_one_metadata_operation_per_group() {
        let (mut slot, _rx) = create_test_slot_with_groups(96, 0, 2);

        slot.onboard_blocks(
            Box::new(TestAnyBlocks {
                ids: vec![500, 501],
                pool: BlockTransferPool::Host,
            }),
            vec![0, 1],
        )
        .unwrap();

        let ops = slot.take_pending_operations().expect("pending ops");
        assert_eq!(ops.len(), 2, "one metadata op per group leg");
        for (group, op) in ops.iter().enumerate() {
            assert_eq!(op.request_id, "test-req");
            assert_eq!(op.transfer_type, TransferType::Load);
            // Immediate ops: the leader's metadata counts them per request
            // into expected_immediate_ops, and the worker's slot completion
            // counts ops per request — so the onboard finishes only when
            // ALL legs did.
            assert_eq!(op.request_type, RequestType::Immediate);
            assert_eq!(op.uuid, derive_leg_uuid(ops[0].uuid, group));
        }
        assert_ne!(ops[0].uuid, ops[1].uuid);
    }
}

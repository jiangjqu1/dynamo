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

/// DYN_KVBM_MAMBA_GROUPS ("0,1,2") + DYN_KVBM_MAMBA_BLOCK_TOKENS (e.g.
/// 1960 — the fp4 GDN state-checkpoint cadence). BOTH must be set: either
/// missing means the hybrid multi-turn continuation contract is OFF (the
/// Python gate refuses multi-group external matches, and the slots never
/// plan a mamba restore). Read once at slot-manager construction — the
/// same two env vars gate the Python side.
fn read_mamba_config_from_env() -> (Vec<usize>, usize) {
    let groups = std::env::var("DYN_KVBM_MAMBA_GROUPS").ok().and_then(|s| {
        let parsed: Option<Vec<usize>> = s
            .split(',')
            .map(|t| t.trim().parse::<usize>().ok())
            .collect();
        parsed.filter(|v| !v.is_empty())
    });
    let tokens = std::env::var("DYN_KVBM_MAMBA_BLOCK_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&t| t > 0);
    match (groups, tokens) {
        (Some(groups), Some(tokens)) => (groups, tokens),
        _ => (Vec::new(), 0),
    }
}

/// Phase A chain-end registry: the hash of the LAST block stored by each
/// finished multi-group request's flush. A multi-group external match is
/// valid ONLY when its end lands on one of these — the tier-2 content at a
/// chain end carries the mamba checkpoint AT that boundary (the storing
/// request's FINAL state); a mid-chain end has no checkpoint and would
/// restore wrong bytes. Populated by `take_hybrid_flush_offload`;
/// consulted by `acquire_local_matches`. The cap guards only memory — an
/// entry's absence after a clear simply refuses matches until the next
/// finish re-records (recompute is always safe).
static HYBRID_CHAIN_END_HASHES: std::sync::LazyLock<std::sync::Mutex<HashSet<SequenceHash>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(HashSet::new()));

fn record_hybrid_chain_end(hash: SequenceHash) {
    let mut set = HYBRID_CHAIN_END_HASHES.lock().unwrap();
    if set.len() > 1_000_000 {
        set.clear();
    }
    set.insert(hash);
}

fn is_hybrid_chain_end(hash: SequenceHash) -> bool {
    HYBRID_CHAIN_END_HASHES.lock().unwrap().contains(&hash)
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

    /// REPLACE the per-group device block-id mapping with the FULL per-group
    /// lists vLLM reports for the request (`per_group[g]` = group `g`'s ids
    /// for the request's logical blocks, position-indexed from block 0).
    ///
    /// vLLM hands full lists at update_state_after_alloc, for new requests,
    /// for resumed-from-preemption requests, and at finish.
    fn set_per_group_device_blocks(
        &mut self,
        per_group_block_ids: &[Vec<BlockId>],
    ) -> Result<(), SlotError>;

    /// APPEND the per-group device block-id DELTAS (this iteration's newly
    /// allocated ids per group — vLLM's cached-request `new_block_ids`) to
    /// the tail of the mapping.
    fn append_per_group_device_blocks(
        &mut self,
        per_group_block_ids: &[Vec<BlockId>],
    ) -> Result<(), SlotError>;

    /// Store the request's per-group onboard destination geometry (fix 2,
    /// HMA multi-group): `per_group_starts[g]`/`per_group_ext_counts[g]`
    /// describe group g's external blocks at the TAIL of ITS OWN computed
    /// range — the vLLM reference's (`SimpleCPUOffloadScheduler::
    /// update_state_after_alloc`) arithmetic, which the flat
    /// evaluated-blocks position cannot express for coarse-granularity
    /// groups. Consumed by the next `trigger_onboarding`; absent keeps
    /// the positional fallback.
    fn set_onboard_geometry(
        &mut self,
        per_group_starts: &[usize],
        per_group_ext_counts: &[usize],
    ) -> Result<(), SlotError>;
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
    /// Phase A: the mamba restore geometry (see the slot field docs),
    /// stamped into every slot at creation.
    mamba_groups: Vec<usize>,
    mamba_block_tokens: usize,
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
        // Phase A: the hybrid mamba restore geometry (see the slot field
        // docs) — the same env pattern as the offload priority above; the
        // restart scripts own the model's constants, and the Python gate
        // reads the SAME two env vars so both sides agree.
        let (mamba_groups, mamba_block_tokens) = read_mamba_config_from_env();
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
            mamba_groups,
            mamba_block_tokens,
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
            self.mamba_groups.clone(),
            self.mamba_block_tokens,
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

    /// Per-group device block ids for HMA multi-group deployments:
    /// `table[g][p]` = group `g`'s device id for the request's p-th logical
    /// block. vLLM's hybrid allocator assigns block ids PER GROUP
    /// independently (the HMA-PROBE logged `[[1], [2], [3], [4]]` for one
    /// request), so any transfer touching group `g` needs group `g`'s OWN
    /// ids — the flat `device_blocks` list is group 0's view and is only
    /// valid for group 0's pool.
    ///
    /// [`Self::set_per_group_device_blocks`] REPLACES the table with the
    /// FULL per-group lists vLLM reports (update_state_after_alloc, new
    /// requests, resumed-from-preemption requests, finish);
    /// [`Self::append_per_group_device_blocks`] extends the tail with the
    /// per-group DELTAS vLLM reports for running requests (cached
    /// `new_block_ids`). `None` until the first report — single-group
    /// deployments never set it and keep the legacy flat path.
    ///
    /// Deliberately NOT cleared by `reset_after_preemption`: the Python
    /// leader re-publishes the full mapping (update_state_after_alloc and
    /// the resumed request's full `new_block_ids`) BEFORE the Rust-side
    /// reset runs inside `build_connector_metadata`, so clearing here would
    /// drop the fresh mapping and leave multi-group consumers with nothing
    /// to index. Between preemption and resume no consumer runs (the
    /// request is unscheduled).
    per_group_block_table: Option<Vec<Vec<BlockId>>>,

    /// Per-group onboard destination geometry published by the Python
    /// leader (fix 2, HMA multi-group): group g's external blocks sit at
    /// the TAIL of ITS OWN computed range
    /// `starts[g] .. starts[g] + ext_counts[g]` — the vLLM reference's
    /// (`SimpleCPUOffloadScheduler::update_state_after_alloc`)
    /// arithmetic, which a flat `evaluated_blocks` position cannot express
    /// for coarse-granularity groups. `None` (single-group, or the Python
    /// side could not derive it) keeps the positional fallback
    /// (`per_group_device_ids(evaluated_blocks, n)`). Consumed by the
    /// next `trigger_onboarding`; the Python side republishes before
    /// every `update_state_after_alloc` that triggers an onboard.
    onboard_geometry: Option<OnboardGeometry>,

    /// Phase A (the hybrid multi-turn continuation contract): the mamba
    /// group indices and the TRUE mamba block size in tokens — the ENGINE's
    /// state-checkpoint cadence (1,960 for the fp4's GDN groups), never the
    /// metadata-uniform block size the factory config reports. Configured
    /// via DYN_KVBM_MAMBA_GROUPS ("0,1,2") + DYN_KVBM_MAMBA_BLOCK_TOKENS;
    /// both required — either missing means the contract is OFF (the Python
    /// gate refuses multi-group external matches and the slots never plan a
    /// mamba restore).
    mamba_groups: Vec<usize>,
    mamba_block_tokens: usize,

    /// Single-group only: the under-coverage WARN fired once for this
    /// request (the chunked-prefill COW transient bursts per step; one
    /// warning carries the whole story — the 2026-10-24 log-flood fix).
    under_cover_warned: bool,
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
        mamba_groups: Vec<usize>,
        mamba_block_tokens: usize,
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
            per_group_block_table: None,
            onboard_geometry: None,
            mamba_groups,
            mamba_block_tokens,
            under_cover_warned: false,
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
        mamba_groups: Vec<usize>,
        mamba_block_tokens: usize,
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
            per_group_block_table: None,
            onboard_geometry: None,
            mamba_groups,
            mamba_block_tokens,
            under_cover_warned: false,
        }
    }

    #[cfg(test)]
    fn device_blocks_snapshot(&self) -> &[BlockId] {
        &self.device_blocks
    }

    #[cfg(test)]
    fn per_group_block_table_snapshot(&self) -> Option<&[Vec<BlockId>]> {
        self.per_group_block_table.as_deref()
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

/// Reconcile vLLM's per-step device-block feed against one tracked
/// block-id sequence (the flat group-0 list, or one HMA group's row in
/// the per-group table). vLLM's feed is authoritative for the request's
/// CURRENT blocks.
///
/// Decision ladder (order matters — vLLM's HMA mamba groups legitimately
/// report DUPLICATE block ids, e.g. `[0, 0, 0, 0, 11, 18]`, so any
/// id-search must come after the alignment-exact cases):
///
/// 1. feed == tracked → no-op;
/// 2. the feed matches the tracked list's natural suffix → no-op (the
///    feed is already applied);
/// 3. the tracked list is a prefix of the feed → extend with the tail.
///
/// Then the id-search cases:
///
/// 4. feed head not tracked → plain append of genuinely new blocks;
/// 5. feed head tracked and the tracked suffix matches the feed's
///    prefix → extend with the non-overlapping tail (overlap dedup);
/// 6. feed head tracked at position 0 and the tracked tail diverged →
///    vLLM reassigned the tracked ids (copy-on-write of a
///    prefix-cache-shared block: its kv_cache_manager swaps
///    `req_blocks[block_idx]` for a private `cow_block`, so a
///    mid-list id the slot still tracks is no longer the request's).
///    The feed is the request's current list — REPLACE the tracked
///    list wholesale;
/// 7. feed head tracked mid-list with a divergent continuation →
///    the ladder could not align the feed. vLLM's feed is ALWAYS the
///    authoritative current list — REPLACE with a warn (vLLM's HMA
///    hybrid reality: replacement-shaped feeds on running steps,
///    cached-mamba states collapsing to a shared id, groups not
///    allocating in step on prefix-cache hits). The warn keeps the
///    divergence diagnosable for the DEP's semantic review.
///
/// The reconcile therefore never fails: alignment-exact cases are
/// cheap no-ops/extensions, and everything else degrades to
/// replace-with-warn (vLLM's feed is truth). Already-tracked ids in
/// append/extension tails also only WARN: vLLM's HMA mamba groups
/// legitimately report duplicate ids within one request.
fn reconcile_device_block_feed(
    tracked: &mut Vec<BlockId>,
    feed: &[BlockId],
) -> Result<(), SlotError> {
    if feed.is_empty() {
        return Ok(());
    }

    // Ladder case 1: identical lists — nothing to do. (With duplicate ids
    // in play, any search-based alignment can misfire; the exact match
    // must be decided first.)
    if tracked.as_slice() == feed {
        return Ok(());
    }

    // Ladder case 2: the feed is the tracked list's natural suffix — the
    // feed is already fully applied (a repeated or trimmed publish).
    if tracked.len() >= feed.len() && tracked[tracked.len() - feed.len()..] == feed[..] {
        return Ok(());
    }

    // Ladder case 3: the tracked list is a prefix of the feed — the feed
    // adds new blocks at the tail. Checked before the id search so a
    // duplicated head cannot pull the alignment to a later occurrence.
    if feed.len() > tracked.len() && feed[..tracked.len()] == tracked[..] {
        let new_ids = &feed[tracked.len()..];
        let existing: HashSet<BlockId> = tracked.iter().copied().collect();
        for id in new_ids {
            if existing.contains(id) {
                // vLLM's HMA mamba groups legitimately report duplicate
                // block ids within one request (verified live: [0,0,0,0,
                // 11,18]); a set-membership rejection would misfire there.
                // Warn, not error — the alignment cases above are the
                // real consistency guards.
                tracing::warn!(
                    block_id = id,
                    "device-block feed extends with an already-tracked id \
                     (vLLM duplicate reporting is legal for HMA mamba groups)"
                );
            }
        }
        tracked.extend_from_slice(new_ids);
        tracing::debug!("DEDUP: prefix extend with {} new block_ids", new_ids.len());
        return Ok(());
    }

    let Some(pos) = tracked.iter().rposition(|&id| id == feed[0]) else {
        // Case (a): no overlap — the feed is all-new blocks.
        let existing: HashSet<BlockId> = tracked.iter().copied().collect();
        for id in feed {
            if existing.contains(id) {
                tracing::warn!(
                    block_id = id,
                    "device-block feed appends an already-tracked id \
                     (vLLM duplicate reporting is legal for HMA mamba groups)"
                );
            }
        }
        tracked.extend_from_slice(feed);
        return Ok(());
    };

    let suffix_len = tracked.len() - pos;
    if suffix_len <= feed.len() && tracked[pos..] == feed[..suffix_len] {
        // Case (b): suffix/prefix overlap — extend with the remainder.
        let new_ids = &feed[suffix_len..];
        if !new_ids.is_empty() {
            let existing: HashSet<BlockId> = tracked.iter().copied().collect();
            for id in new_ids {
                if existing.contains(id) {
                    tracing::warn!(
                        block_id = id,
                        "device-block overlap extends with an already-tracked id \
                         (vLLM duplicate reporting is legal for HMA mamba groups)"
                    );
                }
            }
            tracked.extend_from_slice(new_ids);
        }
        tracing::debug!(
            "DEDUP: suffix/prefix overlap of {} block_ids, extended with {} new",
            suffix_len,
            new_ids.len()
        );
        return Ok(());
    }

    if pos == 0 {
        // Case (c): head matches, tail diverged — vLLM reassigned the
        // tracked ids (e.g. COW). The feed is the request's current list;
        // the old tail ids are no longer the request's.
        tracing::warn!(
            old = ?tracked,
            new = ?feed,
            "device blocks reassigned by vLLM (e.g. copy-on-write of a \
             prefix-cache-shared block); replacing the tracked list"
        );
        tracked.clear();
        tracked.extend_from_slice(feed);
        return Ok(());
    }

    // Case 7 (was (d)): mid-list match with a divergent continuation.
    // vLLM's hybrid reality produces feeds the ladder cannot align —
    // replacement-shaped lists on running steps, cached-mamba state
    // blocks that collapse to a shared id (repeated 0s), groups that do
    // not allocate in step on prefix-cache hits. vLLM's feed is ALWAYS
    // the authoritative CURRENT list: replace and warn (the engine
    // must not die on feed-shape surprises; the warn keeps the
    // divergence diagnosable for the DEP's semantic review).
    tracing::warn!(
        head = feed[0],
        pos,
        old = ?tracked,
        new = ?feed,
        "device-block feed could not be aligned (vLLM HMA hybrid feed \
         shape); replacing the tracked list with the authoritative feed"
    );
    tracked.clear();
    tracked.extend_from_slice(feed);
    Ok(())
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
        // Consumed by the last trigger_onboarding; cleared defensively so
        // a stale geometry can never outlive the request's tracking state.
        self.onboard_geometry = None;
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
        // Hybrid multi-group (fix 1a): evaluated_blocks stays FROZEN here —
        // the mid-flight offload below is suppressed and the finish flush
        // (take_hybrid_flush_offload) must still see every completed
        // position as a candidate. Only trigger_onboarding (which consumes
        // the onboarded prefix) and the flush itself may advance it.
        // Attention-only keeps the validated cursor advance.
        if self.num_groups <= 1 {
            self.evaluated_blocks =
                max(self.evaluated_blocks, num_computed_tokens / self.block_size);
        }

        // Apply new block_ids with the reconciliation contract — see
        // `reconcile_device_block_feed` for the cases: plain append,
        // suffix/prefix overlap dedup, COW-style head-match replacement,
        // and loud failure for inconsistent feeds.
        if !block_ids.is_empty()
            && let Err(e) = reconcile_device_block_feed(&mut self.device_blocks, block_ids)
        {
            panic!("{}", e);
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

        // The tracked list should cover the newly scheduled tokens, but the
        // check is TOLERANT, not fatal: chunked prefill + prefix-cache COW
        // breaks this cadence transiently. On the step vLLM copy-on-writes a
        // shared tail block it can deliver a replacement-shaped feed (live
        // cm008 evidence: a non-resumed step fed [0, 6] where 6 replaced a
        // mid-list id) and the tracked list under-covers the tokens scheduled
        // so far (`next_position: 3136 > device_blocks.len() 2 * block_size
        // 784` — EngineCore death). The python leader now publishes the
        // cumulative full lists every step, and the offload candidate
        // selection below clamps to the tracked list; the next pure-append
        // step restores coverage. Genuine feed inconsistency still fails
        // loudly in `reconcile_device_block_feed`.
        //
        // EMISSION POLICY (2026-10-24, after the ladder-log flood): for
        // MULTI-GROUP (HMA) slots the under-coverage is STRUCTURAL, not
        // transient — `device_blocks` is the flat list = group 0's blocks,
        // and group 0 is a mamba group on the fp4 (a checkpoint-cadence row
        // of ~cdiv(len/1960)+2 entries) while the math divides by the
        // attention block size, so every request beyond ~2.6K tokens trips
        // this on every step; the clamp is inert anyway (fix 1a suppressed
        // multi-group mid-flight offloads — the finish-flush owns the
        // store). Log at DEBUG there. For SINGLE-GROUP slots the case is a
        // genuine one-step COW transient worth one WARN — rate-limited to
        // the first occurrence per request (the bursts fire per step).
        let next_position = self.current_position + num_scheduled_tokens;
        if next_position > self.device_blocks.len() * self.block_size {
            if self.num_groups > 1 {
                tracing::debug!(
                    request_id = %self.request_id,
                    next_position,
                    tracked_blocks = self.device_blocks.len(),
                    block_size = self.block_size,
                    "tracked device blocks under-cover the scheduled tokens \
                     (structural for HMA: the flat list is group 0's mamba \
                     row); clamping offload candidates to the tracked list"
                );
            } else if !self.under_cover_warned {
                self.under_cover_warned = true;
                tracing::warn!(
                    request_id = %self.request_id,
                    next_position,
                    tracked_blocks = self.device_blocks.len(),
                    block_size = self.block_size,
                    "tracked device blocks under-cover the scheduled tokens \
                     (chunked-prefill COW transient, once per request); clamping \
                     offload candidates to the tracked list"
                );
            }
        }

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

        // Clamp the candidate window to the tracked list: a chunked-prefill
        // COW step can leave the list transiently short of the scheduled
        // tokens (see the under-coverage warning above). Candidates beyond
        // the tracked entries do not exist yet — the next pure-append step
        // restores coverage and the deferred blocks offload then. Clamping
        // keeps the position-indexed slice inside real entries, so a stale
        // or missing position can never be offloaded.
        let available_candidates = self
            .device_blocks
            .len()
            .saturating_sub(self.evaluated_blocks);
        let num_candidate_blocks =
            ((next_position / self.block_size) - self.evaluated_blocks).min(available_candidates);

        tracing::debug!(
            "evaluating policy with the following parameters: state: {:?}; current_position: {}; num_candidate_blocks: {}; num_scheduled_tokens: {}",
            self.state,
            self.current_position,
            num_candidate_blocks,
            num_scheduled_tokens
        );

        // Hybrid mid-flight offload suppression (fix 1a): the fp4 COWs its
        // chunk-boundary block on EVERY chunked-prefill boundary (live
        // cm008 evidence: the tracked ids walk [0,1]→[0,6]→[0,11]→
        // [0,16] while the same hash stays matched), so a mid-flight copy
        // captures the PRE-WRITE content of a soon-reassigned id, and BOTH
        // the stale copy and a later final copy would match the same
        // sequence hash — the host pool must contain only final copies.
        // Skip creating the mid-flight op entirely: do not advance
        // evaluated_blocks (frozen above); leave the candidates for the
        // finish flush (take_hybrid_flush_offload), which runs after the
        // Python finish-publish has settled the per-group table.
        // Attention-only (num_groups <= 1) is byte-identical: its blocks
        // are never reassigned mid-flight, and the validated coder path
        // runs unchanged below.
        if self.num_groups > 1 {
            tracing::debug!(
                request_id = %self.request_id,
                num_candidate_blocks,
                "hybrid multi-group request: mid-flight offload suppressed; the finish flush owns the candidates"
            );
        } else if num_candidate_blocks != 0 {
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

            // num_candidate_blocks is clamped to the tracked entries above, so
            // the slice below always yields exactly that many candidates — the
            // former "device block overflow" assert is unreachable.

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

        // A previous onboard is still in flight (vLLM's multi-step
        // onboarding lifecycle re-queries get_num_new_matched_tokens on
        // later scheduler steps — the 2026-10-24 ladder fatal: "slot must
        // be in the NotScheduled or Preempted state ...; got
        // Onboarding(14896)" killed EngineCore mid-run). The in-flight match
        // completes on its own; this step reports no NEW external tokens.
        if matches!(self.state(), SlotState::Onboarding(_)) {
            tracing::debug!(
                request_id = %self.request_id,
                state = ?self.state(),
                "onboard in flight; skipping this lookup step"
            );
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

        let num_matched_blocks = num_matched_host_blocks + num_matched_disk_blocks;

        // TEMPORARY MATCH-PROBE (debug level — enable via
        // DYN_LOG=info,_core::block_manager=debug; remove after the H2D
        // validation lands): the pinned-baseline external lookup returned
        // zero hits with P's blocks stored.
        tracing::debug!(
            request_id = %self.request_id,
            queried = blocks_to_lookup.len(),
            search_offset,
            first_hash = ?sequence_hashes.first().map(|h| h.to_string()),
            matched_host = num_matched_host_blocks,
            matched_disk = num_matched_disk_blocks,
            "MATCH-PROBE: external lookup result"
        );

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

        // The matched region is block-quantized; the request's token count
        // need not be (chat templates leave a sub-block tail, so a
        // full-history match can land up to one block PAST total_tokens).
        // Drop the last matched block in both cases: on the exact boundary
        // the final block stays locally computed, and on the overshoot the
        // tail is a partial block that can never be onboarded. (The
        // 2026-10-22 EngineCore fatal — "cannot advance computed position
        // ... total tokens is ..." on every prefix-shared reuse turn —
        // was the missing overshoot arm.)
        if (num_computed_tokens + num_new_matched_tokens) >= self.sequence().total_tokens() {
            tracing::debug!(
                "matched region reaches the request end (boundary or past the unaligned tail); throwing away the last block"
            );

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

        // Phase A (the hybrid multi-turn continuation contract): a
        // MULTI-GROUP external match is valid ONLY when its end lands on a
        // stored chain end — the tier-2 block there carries the mamba
        // checkpoint AT that boundary (the storing request's FINAL state);
        // every other end has no checkpoint and would restore wrong bytes.
        // Refuse (recompute is always correct) — the lookup itself stays
        // counted so the hit-rate metrics reflect the refusal honestly.
        if self.num_groups > 1 {
            let staged_blocks = host_blocks.len() + disk_blocks.len();
            let match_end_hash = sequence_hashes[num_computed_blocks + staged_blocks - 1];
            if !is_hybrid_chain_end(match_end_hash) {
                tracing::debug!(
                    request_id = %self.request_id,
                    matched = staged_blocks,
                    "multi-group match does not end at a stored chain end \
                     (no mamba checkpoint at the boundary); refusing — recompute"
                );
                return Ok(());
            }
        }

        // Count the served tokens only now, post-validation: a refused match
        // must not inflate the hit-rate metrics.
        self.record_cached_host_tokens(host_blocks.len() * block_size);
        self.record_cached_disk_tokens(disk_blocks.len() * block_size);

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

        // The staged/external count was computed at LOOKUP time against
        // the then-current computed position; vLLM can advance the
        // position before this trigger (prefill progress registered
        // between the two hooks), so the block-quantized count can now
        // overshoot the request's unaligned tail by up to a block — the
        // 2026-10-22 EngineCore fatal ("cannot advance computed position
        // ... total tokens is ..." on every prefix-shared reuse turn).
        // Same treatment as the lookup-time boundary case: drop the
        // last staged block and clamp the advance to the whole blocks
        // that still fit — the sub-block tail is locally computed anyway
        // (a partial block can never be onboarded).
        let mut num_external_tokens = num_external_tokens;
        if self.current_position + num_external_tokens > self.sequence().total_tokens() {
            let fitting = self.sequence().total_tokens() - self.current_position;
            let clamped = fitting - (fitting % self.block_size);
            tracing::debug!(
                request_id = %self.request_id,
                "clamping external advance from {} to {} tokens (computed position advanced since the lookup; the tail stays locally computed)",
                num_external_tokens, clamped
            );
            if clamped == 0 {
                self.staging_from_host = None;
                self.staging_from_disk = None;
            } else if let Some(blocks) = self.staging_from_disk.as_mut() {
                blocks.pop();
            } else if let Some(blocks) = self.staging_from_host.as_mut() {
                blocks.pop();
            }
            num_external_tokens = clamped;
        }

        let num_computed_blocks = self.current_position / self.block_size;

        // shift the evaluated blocks position to the end of the computed/cached blocks
        self.evaluated_blocks = num_computed_blocks;

        // Fix 2: consume the per-group onboard geometry (published by the
        // Python leader for multi-group requests before this trigger) —
        // each group's destinations then come from its OWN row's tail
        // slice instead of the flat evaluated-blocks position. Absent
        // (single-group, or the Python side could not derive it): the
        // positional fallback, exactly today's behavior.
        let onboard_geometry = self.onboard_geometry.take();

        // Phase A (the hybrid multi-turn continuation contract): plan the
        // mamba restore — vLLM's align-mode engine reads the resumed state
        // from row position (P-1)/mamba_block_tokens (preprocess_mamba),
        // and the checkpoint content (the storing request's FINAL state)
        // lives in the LAST staged tier-2 block, the chain end the match
        // was validated against. The plan applies to the op carrying the
        // last staged block.
        let mamba_plan = self.build_mamba_restore_plan(num_external_tokens)?;

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

            // The mamba plan belongs to the LAST staged block: the disk op
            // carries it when disk blocks exist, else this host op does.
            let mamba_for_op = if self.staging_from_disk.is_some() {
                None
            } else {
                mamba_plan.as_ref()
            };
            self.onboard_blocks(
                src_blocks,
                dst_block_ids,
                onboard_geometry.as_ref(),
                mamba_for_op,
            )?;

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

            // The disk op (when it exists) carries the last staged block —
            // the mamba plan's checkpoint source.
            self.onboard_blocks(
                src_blocks,
                dst_block_ids,
                onboard_geometry.as_ref(),
                mamba_plan.as_ref(),
            )?;

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

    fn set_per_group_device_blocks(
        &mut self,
        per_group_block_ids: &[Vec<BlockId>],
    ) -> Result<(), SlotError> {
        Self::validate_per_group_shape(per_group_block_ids, self.num_groups)?;
        tracing::debug!(
            request_id = self.request_id,
            num_groups = per_group_block_ids.len(),
            positions = per_group_block_ids.first().map(Vec::len).unwrap_or(0),
            "replacing per-group device block mapping (full lists)"
        );
        self.per_group_block_table = Some(per_group_block_ids.to_vec());
        Ok(())
    }

    fn append_per_group_device_blocks(
        &mut self,
        per_group_block_ids: &[Vec<BlockId>],
    ) -> Result<(), SlotError> {
        Self::validate_per_group_shape(per_group_block_ids, self.num_groups)?;
        let table = self
            .per_group_block_table
            .get_or_insert_with(|| vec![Vec::new(); self.num_groups]);
        for (group, (group_ids, delta)) in table.iter_mut().zip(per_group_block_ids).enumerate() {
            // Same reconciliation as the flat list: plain append, overlap
            // dedup, and COW-style head-match replacement. The table is
            // advisory, so an inconsistent feed (mid-list divergence) is
            // logged and skips the row update instead of panicking — the
            // flat apply enforces that contract; consumers fail loudly if
            // they later need a stale position.
            if let Err(e) = reconcile_device_block_feed(group_ids, delta) {
                tracing::error!(
                    request_id = %self.request_id,
                    group = group,
                    "skipping inconsistent per-group device block delta: {}",
                    e
                );
            }
        }
        tracing::debug!(
            request_id = self.request_id,
            num_groups = per_group_block_ids.len(),
            delta_len = per_group_block_ids.first().map(Vec::len).unwrap_or(0),
            total = table.first().map(Vec::len).unwrap_or(0),
            "appended per-group device block delta to mapping"
        );
        Ok(())
    }

    fn set_onboard_geometry(
        &mut self,
        per_group_starts: &[usize],
        per_group_ext_counts: &[usize],
    ) -> Result<(), SlotError> {
        if per_group_starts.len() != self.num_groups
            || per_group_ext_counts.len() != self.num_groups
        {
            return Err(SlotError::InvalidOperation(format!(
                "onboard geometry for request {} expects {} per-group entries, \
                 got starts={} ext_counts={}",
                self.request_id,
                self.num_groups,
                per_group_starts.len(),
                per_group_ext_counts.len()
            )));
        }
        tracing::debug!(
            request_id = self.request_id,
            starts = ?per_group_starts,
            ext_counts = ?per_group_ext_counts,
            "storing per-group onboard destination geometry"
        );
        self.onboard_geometry = Some(OnboardGeometry {
            starts: per_group_starts.to_vec(),
            ext_counts: per_group_ext_counts.to_vec(),
        });
        Ok(())
    }
}

impl VllmConnectorSlot {
    /// Phase A: plan the mamba groups' restore for the onboard. The
    /// external range ends at P = current_position + num_external_tokens
    /// tokens; vLLM's align-mode engine reads the resumed state from row
    /// position (P-1)/mamba_block_tokens (preprocess_mamba's
    /// `prev_state_idx`), and the checkpoint content — the storing
    /// request's FINAL state — lives in the LAST staged tier-2 block (the
    /// chain end the match was validated against). Returns Ok(None) when
    /// the contract is unconfigured or nothing is staged; errors LOUDLY
    /// when a mamba row cannot host the engine's state slot — restoring
    /// wrong bytes is never acceptable, and recompute is no longer
    /// possible past the match (the tokens were already promised).
    fn build_mamba_restore_plan(
        &self,
        num_external_tokens: usize,
    ) -> Result<Option<MambaRestorePlan>, SlotError> {
        if self.mamba_groups.is_empty() {
            return Ok(None);
        }
        if self.staging_from_host.is_none() && self.staging_from_disk.is_none() {
            return Ok(None);
        }
        let p_tokens = self.current_position + num_external_tokens;
        if p_tokens == 0 {
            return Ok(None);
        }
        let state_slot = (p_tokens - 1) / self.mamba_block_tokens;
        let Some(table) = &self.per_group_block_table else {
            return Err(SlotError::InvalidOperation(format!(
                "multi-group onboard for request {} without a per-group device table",
                self.request_id
            )));
        };
        let mut dst_per_group = HashMap::new();
        for &group in &self.mamba_groups {
            let row = table.get(group).ok_or_else(|| {
                SlotError::InvalidOperation(format!(
                    "per-group table of request {} is missing group {group}",
                    self.request_id
                ))
            })?;
            if row.len() <= state_slot {
                return Err(SlotError::InvalidOperation(format!(
                    "mamba row of group {group} for request {} does not cover the \
                     engine's state slot {state_slot} (P={p_tokens} tokens, row \
                     length {}): refusing to restore wrong bytes",
                    self.request_id,
                    row.len()
                )));
            }
            dst_per_group.insert(group, row[state_slot]);
        }
        tracing::debug!(
            request_id = %self.request_id,
            p_tokens,
            state_slot,
            groups = ?self.mamba_groups,
            "Phase A mamba restore planned (state-slot targets)"
        );
        Ok(Some(MambaRestorePlan { dst_per_group }))
    }

    /// Shared shape validation for per-group mapping updates: exactly one
    /// list per HMA group. Group lists of UNEQUAL length are logged at
    /// DEBUG: the 2026-10-22 store-model pin proved divergent lengths are
    /// the EXPECTED growing state (mamba rows are checkpoint-cadence while
    /// attention rows are position-indexed — 6,134 WARNs per ladder cell
    /// before the demotion); the consumers fail loudly if a needed
    /// position is missing.
    fn validate_per_group_shape(
        per_group_block_ids: &[Vec<BlockId>],
        num_groups: usize,
    ) -> Result<(), SlotError> {
        if per_group_block_ids.len() != num_groups {
            return Err(SlotError::InvalidOperation(format!(
                "per-group device block mapping expects {num_groups} group lists, got {}",
                per_group_block_ids.len()
            )));
        }
        if num_groups > 1 {
            let lengths: Vec<usize> = per_group_block_ids.iter().map(Vec::len).collect();
            if lengths.windows(2).any(|w| w[0] != w[1]) {
                tracing::debug!(
                    group_lengths = ?lengths,
                    "per-group device block lists have unequal lengths (expected: the \
                     mamba rows grow at checkpoint cadence, attention at position cadence)"
                );
            }
        }
        Ok(())
    }

    /// Each group's device ids for logical block positions `start..start+len`.
    ///
    /// Returns `Ok(None)` for the legacy single-group path (one group, or
    /// the mapping was never reported) — callers then use group 0's flat
    /// ids for every group. Returns an error when the slot is multi-group
    /// but the mapping is missing or too short: moving group 0's ids
    /// through another group's pool would transfer the WRONG bytes, so the
    /// op must not be issued.
    fn per_group_device_ids(
        &self,
        start: usize,
        len: usize,
    ) -> Result<Option<Vec<Vec<BlockId>>>, SlotError> {
        if self.num_groups <= 1 {
            return Ok(None);
        }
        let Some(table) = &self.per_group_block_table else {
            return Err(SlotError::InvalidOperation(format!(
                "per-group device block mapping missing for request {} (num_groups={}): \
                 vLLM's per-group ids were never reported; refusing to move group 0's ids \
                 through every group's pool",
                self.request_id, self.num_groups
            )));
        };
        let end = start + len;
        let lengths: Vec<usize> = table.iter().map(Vec::len).collect();
        if lengths.iter().any(|&l| l < end) {
            return Err(SlotError::InvalidOperation(format!(
                "per-group device block mapping too short for request {}: need positions \
                 {start}..{end}, group lengths {lengths:?}",
                self.request_id
            )));
        }
        Ok(Some(
            (0..self.num_groups)
                .map(|g| table[g][start..end].to_vec())
                .collect(),
        ))
    }

    /// Finish-time offload for hybrid requests (fix 1b): the remaining
    /// completed positions, using the CURRENT (final) per-group ids.
    /// Called from the leader's `request_finished` AFTER the Python
    /// finish-publish refreshed the per-group table (the COW churn has
    /// settled; the content is final).
    ///
    /// Multi-group only — the mid-flight offload is suppressed (fix 1a),
    /// so every mid-flight candidate is still pending here and
    /// `evaluated_blocks` was frozen (only `trigger_onboarding`, which
    /// consumes the onboarded prefix, advanced it). Attention-only slots
    /// keep the validated mid-flight path and return `None`.
    ///
    /// Sends the [`LocalTransferRequest`] on the slot's transfer channel
    /// IMMEDIATELY — the copy must execute right away, racing only the
    /// documented small free-window like the attention-only path — and
    /// RETURNS the per-leg [`WorkerTransferRequest`]s for the leader to
    /// defer into the next `build_connector_metadata`: the request is no
    /// longer scheduled, so the slot's pending-operations drain would
    /// never reach them.
    ///
    /// This is the designated finish path: it deliberately does NOT go
    /// through [`Self::offload_blocks`], whose "Finishing blocks new op
    /// creation" guard does not apply here. A missing/short per-group
    /// mapping errors, exactly like the mid-flight path: moving group
    /// 0's ids through another group's pool transfers the wrong bytes.
    pub(crate) fn take_hybrid_flush_offload(
        &mut self,
    ) -> Result<Option<Vec<WorkerTransferRequest>>, SlotError> {
        // TEMPORARY FLUSH-PROBE (debug level — enable via
        // DYN_LOG=info,_core::block_manager=debug; remove after the H2D fix
        // validation):
        tracing::debug!(
            request_id = %self.request_id,
            num_groups = self.num_groups,
            state = ?self.state,
            evaluated_blocks = self.evaluated_blocks,
            current_position = self.current_position,
            block_size = self.block_size,
            "FLUSH-PROBE: take_hybrid_flush_offload entered"
        );
        if self.num_groups <= 1 {
            return Ok(None);
        }
        debug_assert!(
            matches!(self.state, SlotState::Finishing | SlotState::Finished),
            "the finish flush runs after mark_as_finished; got {:?}",
            self.state
        );

        // evaluated_blocks was frozen for hybrid since admission (fix 1a);
        // only trigger_onboarding advanced it past the onboarded prefix.
        let start = self.evaluated_blocks;
        let total_completed = self.current_position / self.block_size;
        let count = total_completed - start;
        if count == 0 {
            return Ok(None);
        }

        // The CURRENT (final) per-group ids for the flush positions; the
        // group-0 flat list for the op is the same table's row 0 (the
        // authoritative final ids — the flat device_blocks list can lag a
        // transient COW step, the just-refreshed table cannot).
        let Some(per_group_block_ids) = self.per_group_device_ids(start, count)? else {
            return Ok(None); // unreachable for num_groups > 1
        };
        let block_ids = per_group_block_ids.first().cloned().unwrap_or_default();

        let token_blocks: Vec<TokenBlock> = self
            .sequence
            .blocks()
            .iter()
            .skip(start)
            .take(count)
            .cloned()
            .collect();

        // Phase A: record the chain end — the tier-2 block at this position
        // carries this request's FINAL mamba state (the checkpoint AT this
        // boundary), the only mamba content a multi-group restore may ever
        // move. A later multi-group match ending HERE is valid; anything
        // shorter has no checkpoint and is refused by acquire_local_matches.
        if let Some(last) = token_blocks.last() {
            record_hybrid_chain_end(last.sequence_hash());
        }
        // Priorities: the stored per-block map (populated by the scheduler
        // feeds), default 0 where absent — the same lookup the mid-flight
        // path performs.
        let priorities: Vec<u32> = block_ids
            .iter()
            .map(|id| self.stored_block_priorities.get(id).copied().unwrap_or(0))
            .collect();

        let operation_id = uuid::Uuid::new_v4();

        let xfer_req = LocalTransferRequest::Offload(LocalOffloadRequest::new(
            self.request_id.clone(),
            block_ids,
            token_blocks,
            priorities,
            operation_id,
            per_group_block_ids,
        ));

        // ONE metadata operation PER GROUP LEG with its deterministic
        // per-leg uuid (the Task-4 pattern): the transfer engine stamps
        // the same uuids on the wire legs, and the worker-side
        // rendezvous — one Worker + one Transfer per (request_id, uuid)
        // — requires the 1:1 match. RequestType::Flush: the worker
        // enqueues these at bind time (no forward pass remains to gate
        // them — a Scheduled hold would deadlock the parked transfer on
        // the idle step after the slot's vacuous reap).
        let mut operations = Vec::with_capacity(self.num_groups);
        for group in 0..self.num_groups {
            operations.push(WorkerTransferRequest {
                request_id: self.request_id.clone(),
                uuid: derive_leg_uuid(operation_id, group),
                transfer_type: TransferType::Store,
                request_type: RequestType::Flush,
            });
        }

        // The transfer executes right away — same channel, same engine as
        // offload_blocks.
        if let Err(e) = self.xfer_tx.send(xfer_req) {
            tracing::error!("Failed to send transfer request: {:?}", e);
            return Err(SlotError::InvalidOperation(format!(
                "Transfer engine unavailable: {}; aborting hybrid finish flush",
                e
            )));
        }

        // The flush consumed the candidates.
        self.evaluated_blocks += count;

        tracing::debug!(
            request_id = self.request_id,
            operation_id = %operation_id,
            legs = self.num_groups,
            start,
            count,
            "hybrid finish flush offload issued"
        );

        Ok(Some(operations))
    }

    /// Per-group onboard DESTINATION rows under the published geometry
    /// (fix 2): group g's row is
    /// `table[g][starts[g] .. starts[g] + ext_counts[g]]` — the vLLM
    /// reference's per-group tail arithmetic (a coarse group's external
    /// tail is NOT the flat evaluated-blocks position). Fails loudly on
    /// a missing or short mapping, mirroring [`Self::per_group_device_ids`]:
    /// moving group 0's ids through another group's pool transfers the
    /// wrong bytes, so the op must not be issued.
    fn onboard_geometry_dst_rows(
        &self,
        geometry: &OnboardGeometry,
    ) -> Result<Vec<Vec<BlockId>>, SlotError> {
        let Some(table) = &self.per_group_block_table else {
            return Err(SlotError::InvalidOperation(format!(
                "per-group device block mapping missing for request {} (num_groups={}): \
                 the onboard geometry needs each group's own destination ids; refusing to \
                 move group 0's ids through every group's pool",
                self.request_id, self.num_groups
            )));
        };
        if geometry.starts.len() < self.num_groups || geometry.ext_counts.len() < self.num_groups {
            return Err(SlotError::InvalidOperation(format!(
                "onboard geometry for request {} covers fewer than {} groups \
                 (starts={}, ext_counts={})",
                self.request_id,
                self.num_groups,
                geometry.starts.len(),
                geometry.ext_counts.len()
            )));
        }
        (0..self.num_groups)
            .map(|g| {
                let start = geometry.starts[g];
                let count = geometry.ext_counts[g];
                let end = start + count;
                let row = &table[g];
                if row.len() < count {
                    return Err(SlotError::InvalidOperation(format!(
                        "per-group device block mapping too short for request {} group {g}: \
                         the onboard geometry needs {count} destination blocks, the group has {}",
                        self.request_id,
                        row.len()
                    )));
                }
                // vLLM HMA groups are not all token-position-indexed: mamba
                // groups report checkpoint-entry rows (shorter than the
                // computed token range — the 2026-10-21 fatal shape: the
                // geometry asked for positions 10..19 of a 10-entry row), and
                // the scheduler may accept fewer blocks than the connector
                // reported (vLLM's own documented case). The external
                // destinations are the row's tail-most `count` entries of the
                // computed range: slice absolutely when the row covers it
                // (token-position groups — unchanged behavior), else fall
                // back row-relatively from the row's own end.
                let slice_start = start.min(row.len() - count);
                if slice_start != start {
                    tracing::warn!(
                        request_id = %self.request_id,
                        group = g,
                        geometry_start = start,
                        geometry_end = end,
                        row_len = row.len(),
                        "onboard geometry position diverges from the group row \
                         (vLLM HMA checkpoint-indexed group); slicing destinations \
                         from the row's tail"
                    );
                }
                Ok(row[slice_start..slice_start + count].to_vec())
            })
            .collect()
    }

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

        // The candidates are the request's logical blocks at positions
        // `evaluated_blocks..+n` (apply_scheduler_output selected them from
        // device_blocks before advancing evaluated_blocks). Resolve each
        // group's SOURCE device ids for exactly those positions — group 0's
        // flat ids are only valid for group 0's pool.
        let per_group_src_block_ids = self
            .per_group_device_ids(self.evaluated_blocks, block_ids.len())?
            .unwrap_or_default();

        let xfer_req = LocalTransferRequest::Offload(LocalOffloadRequest::new(
            self.request_id.clone(),
            block_ids.to_vec(),
            token_blocks.to_vec(),
            priorities.to_vec(),
            operation_id,
            per_group_src_block_ids,
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

    /// Creates the transfer-engine onboard request and the matching
    /// per-leg connector worker requests; they share the logical op uuid
    /// (each leg carries its deterministic `derive_leg_uuid` derivative).
    ///
    /// `onboard_geometry` (fix 2, HMA multi-group): when present, each
    /// group's destinations come from its OWN row's tail slice
    /// `[starts[g] .. starts[g] + ext_counts[g]]` of the per-group table
    /// (the vLLM reference's per-group arithmetic — a coarse group's
    /// external tail is NOT the flat evaluated-blocks position); the
    /// flat `dst_block_ids` (group 0's view) keeps the caller's
    /// positional semantics. When absent: the positional resolution
    /// over [`Self::per_group_device_ids`], exactly today's behavior.
    fn onboard_blocks(
        &mut self,
        src_blocks: Box<dyn AnyBlocks>,
        dst_block_ids: Vec<BlockId>,
        onboard_geometry: Option<&OnboardGeometry>,
        mamba_plan: Option<&MambaRestorePlan>,
    ) -> Result<(), SlotError> {
        debug_assert_eq!(src_blocks.len(), dst_block_ids.len());

        let num_blocks = src_blocks.len();
        let src_storage_pool = src_blocks.storage_pool();
        let operation_id = uuid::Uuid::new_v4();

        let (mut per_group_dst_block_ids, mut per_group_src_offsets) = match onboard_geometry {
            Some(geometry) => (
                self.onboard_geometry_dst_rows(geometry)?,
                vec![0usize; self.num_groups],
            ),
            None => (
                self.per_group_device_ids(self.evaluated_blocks, dst_block_ids.len())?
                    .unwrap_or_default(),
                Vec::new(),
            ),
        };

        // Phase A: the mamba groups' destinations are NOT positional — the
        // engine reads the resumed state from ONE row slot, and the source
        // is the LAST staged block (the chain end carrying the checkpoint).
        // Override those groups' destination rows and source offsets; the
        // attention group keeps the positional slice.
        if let Some(plan) = mamba_plan {
            if per_group_src_offsets.is_empty() && self.num_groups > 1 {
                per_group_src_offsets = vec![0usize; self.num_groups];
            }
            let last_src = num_blocks.saturating_sub(1);
            for (&group, &dst) in &plan.dst_per_group {
                per_group_dst_block_ids[group] = vec![dst];
                per_group_src_offsets[group] = last_src;
            }
        }

        let xfer_req = LocalTransferRequest::Onboard(LocalOnboardRequest::new(
            self.request_id.clone(),
            src_blocks,
            dst_block_ids,
            operation_id,
            per_group_dst_block_ids,
            per_group_src_offsets,
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

/// The per-group onboard destination geometry (fix 2, HMA multi-group):
/// group g's external blocks occupy `starts[g] .. starts[g] + ext_counts[g]`
/// of its OWN computed range — the reference
/// `SimpleCPUOffloadScheduler::update_state_after_alloc` arithmetic (ext
/// blocks at the TAIL, per group; a coarse-granularity group takes FEWER
/// blocks than the scheduler-granularity flat count).
struct OnboardGeometry {
    /// First computed-range position of group g's external blocks.
    starts: Vec<usize>,
    /// How many external blocks group g takes (`n_ext_g`).
    ext_counts: Vec<usize>,
}

/// Phase A (the hybrid multi-turn continuation contract): the mamba groups'
/// restore plan for one onboard op — each mamba group's SINGLE destination
/// row slot (vLLM's align-mode engine reads the resumed state from row
/// position `(P-1)/mamba_block_tokens`; see preprocess_mamba). The source
/// is always the op's LAST staged tier-2 block (the chain end carrying the
/// checkpoint) — wired via the per-group source offsets, not this struct.
struct MambaRestorePlan {
    /// group index → the engine state-slot device id.
    dst_per_group: HashMap<usize, BlockId>,
}

struct LocalOffloadRequest {
    request_id: String,
    /// Group 0's device ids for the candidate blocks — also the logical
    /// block count for the tier-side allocation.
    block_ids: Vec<BlockId>,
    token_blocks: Vec<TokenBlock>,
    /// Priorities for each block, used to set BasicMetadata.priority during offload.
    priorities: Vec<u32>,
    operation_id: uuid::Uuid,
    /// Per-group SOURCE device ids for the candidate blocks, group-major and
    /// position-indexed (`[g][p]` = group `g`'s device id for the p-th
    /// candidate). Empty in the legacy single-group path — every group then
    /// uses `block_ids` (group 0's ids).
    per_group_block_ids: Vec<Vec<BlockId>>,
}

impl LocalOffloadRequest {
    pub fn new(
        request_id: String,
        block_ids: Vec<BlockId>,
        token_blocks: Vec<TokenBlock>,
        priorities: Vec<u32>,
        operation_id: uuid::Uuid,
        per_group_block_ids: Vec<Vec<BlockId>>,
    ) -> Self {
        debug_assert!(block_ids.len() == token_blocks.len());
        debug_assert!(block_ids.len() == priorities.len());
        debug_assert!(
            per_group_block_ids.is_empty() || per_group_block_ids.len() > 1,
            "per-group source ids are only recorded for multi-group offloads"
        );
        for ids in &per_group_block_ids {
            debug_assert_eq!(
                ids.len(),
                block_ids.len(),
                "per-group source ids must be position-aligned with the candidate list"
            );
        }
        Self {
            request_id,
            block_ids,
            token_blocks,
            priorities,
            operation_id,
            per_group_block_ids,
        }
    }
}

struct LocalOnboardRequest {
    request_id: String,
    src_blocks: Box<dyn AnyBlocks>,
    /// Group 0's destination device ids — also the logical block count.
    dst_block_ids: Vec<BlockId>,
    operation_id: uuid::Uuid,
    /// Per-group DESTINATION device ids, group-major (`[g][p]` = group
    /// `g`'s device id for the p-th destination). Empty in the legacy
    /// single-group path — every group then uses `dst_block_ids`. With
    /// the per-group onboard geometry (fix 2), row `g` carries group
    /// `g`'s OWN ext-block count (`ext_counts[g]`): a coarse-granularity
    /// group takes fewer destinations than the flat scheduler-granularity
    /// count, and the transfer pairing truncates to its own count.
    per_group_dst_block_ids: Vec<Vec<BlockId>>,
    /// Per-group SOURCE offsets into the staged tier-2 block list (Phase
    /// A): all-zero (or empty) = identity pairing (today's behavior). The
    /// mamba groups' leg pairs its ONE destination with the LAST staged
    /// block (the chain end carrying the checkpoint), not the first.
    per_group_src_offsets: Vec<usize>,
}

impl LocalOnboardRequest {
    pub fn new(
        request_id: String,
        src_blocks: Box<dyn AnyBlocks>,
        dst_block_ids: Vec<BlockId>,
        operation_id: uuid::Uuid,
        per_group_dst_block_ids: Vec<Vec<BlockId>>,
        per_group_src_offsets: Vec<usize>,
    ) -> Self {
        debug_assert!(src_blocks.len() == dst_block_ids.len());
        debug_assert!(
            per_group_dst_block_ids.is_empty() || per_group_dst_block_ids.len() > 1,
            "per-group destination ids are only recorded for multi-group onboards"
        );
        debug_assert!(
            per_group_src_offsets.is_empty()
                || per_group_src_offsets.len() == per_group_dst_block_ids.len().max(1),
            "per-group source offsets ({}) must match the per-group destination rows ({})",
            per_group_src_offsets.len(),
            per_group_dst_block_ids.len()
        );
        for (ids, offset) in per_group_dst_block_ids
            .iter()
            .zip(per_group_src_offsets.iter().chain(std::iter::repeat(&0)))
        {
            debug_assert!(
                ids.len() <= dst_block_ids.len(),
                "per-group destination ids ({}) exceed the flat destination list ({})",
                ids.len(),
                dst_block_ids.len()
            );
            debug_assert!(
                offset + ids.len() <= src_blocks.len(),
                "per-group source offset ({}) + destinations ({}) exceed the source blocks ({})",
                offset,
                ids.len(),
                src_blocks.len()
            );
        }
        Self {
            request_id,
            src_blocks,
            dst_block_ids,
            operation_id,
            per_group_dst_block_ids,
            per_group_src_offsets,
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
    // 1. Acquire mutable blocks — BEST-EFFORT: the tier-2 pool can be
    // momentarily full under multi-user churn (the finish-flush of several
    // requests landing together). Losing a store only means that prefix
    // is not reusable from tier-2; the request's serving is unaffected
    // (the offload is a copy at rest). Log at WARN and skip — the "No
    // blocks to register" ERROR (2026-10-23/24 runs) polluted the health
    // check and buried real diagnostics for a benign condition.
    let blocks = match storage_pool
        .allocate_blocks(offload_req.block_ids.len())
        .await
    {
        Ok(blocks) => blocks,
        Err(e) => {
            tracing::warn!(
                request_id = %offload_req.request_id,
                requested = offload_req.block_ids.len(),
                pool = storage_name,
                error = %e,
                "tier-2 allocation failed (pool momentarily full); skipping \
                 this store — the prefix is not reusable from tier-2 but \
                 serving is unaffected"
            );
            return Ok(());
        }
    };
    let token_blocks = offload_req.token_blocks;
    let allocated_block_ids: Vec<usize> = blocks.iter().map(|b| b.block_id()).collect();
    let num_blocks = allocated_block_ids.len();

    // TEMPORARY STORE-PROBE (debug level — enable via
    // DYN_LOG=info,_core::block_manager=debug; remove after the H2D
    // validation): the hash chain the offload registers under — paired with
    // the MATCH-PROBE in acquire_local_matches to diagnose zero-hit lookups —
    // plus the allocated tier slots (pairs with the ONBOARD-PROBE).
    tracing::debug!(
        request_id = %offload_req.request_id,
        num_blocks = token_blocks.len(),
        first_hash = ?token_blocks.first().map(|b| b.sequence_hash().to_string()),
        first_ids = ?allocated_block_ids.iter().take(3).collect::<Vec<_>>(),
        "STORE-PROBE: offload registering blocks"
    );

    // Each group moves its OWN source device ids into the SAME freshly
    // allocated tier blocks (the tier pool's ids index every group's
    // worker-side view; only the DEVICE side is per-group). Legacy
    // single-group offloads carry no per-group mapping and use group 0's
    // ids for every group, exactly as before.
    let src_ids_per_group: Vec<Vec<usize>> = if offload_req.per_group_block_ids.is_empty() {
        vec![offload_req.block_ids.clone(); num_groups]
    } else {
        offload_req.per_group_block_ids.clone()
    };
    let block_pairs_per_group: Vec<Vec<(usize, usize)>> = src_ids_per_group
        .into_iter()
        .map(|src_ids| {
            debug_assert_eq!(
                src_ids.len(),
                num_blocks,
                "per-group source ids must cover every candidate block"
            );
            src_ids
                .into_iter()
                .zip(allocated_block_ids.iter().copied())
                .collect()
        })
        .collect();

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
    // GROUP — each leg carries that group's OWN source device ids (the tier
    // destination ids are shared) and its own deterministic uuid. All legs
    // are sent before any is awaited. Single-group (num_groups == 1) sends
    // the one request exactly as before, with the logical op's uuid
    // (derive_leg_uuid identity).
    let notify_receivers = fan_out_group_legs(
        leader,
        num_groups,
        &offload_req.request_id,
        offload_req.operation_id,
        BlockTransferPool::Device,
        transfer_pool,
        &block_pairs_per_group,
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
    let immutable_blocks = match storage_pool.register_blocks(blocks_to_register).await {
        Ok(blocks) => {
            // TEMPORARY STORE-PROBE (debug level — enable via
            // DYN_LOG=info,_core::block_manager=debug; remove after the H2D
            // validation): the registration succeeded — the hashes are now
            // matchable in the pool's registry.
            tracing::debug!(
                request_id = %request_id,
                operation_id = %operation_id,
                registered = blocks.len(),
                "STORE-PROBE: register_blocks OK"
            );
            blocks
        }
        Err(e) => {
            tracing::warn!(
                request_id = %request_id,
                operation_id = %operation_id,
                error = %e,
                "STORE-PROBE: register_blocks FAILED — stored bytes are unmatchable"
            );
            return Err(e.into());
        }
    };

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
/// Each leg carries its own deterministic uuid (`derive_leg_uuid`), its
/// group tag, and ITS GROUP'S OWN block pairs (`block_pairs_per_group[g]` —
/// the per-group device-id mapping; on offloads the sources differ per
/// group, on onboards the destinations do, while the shared tier side is
/// identical) — the worker-side connector scheduler rendezvous keys on
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
    block_pairs_per_group: &[Vec<(usize, usize)>],
    request_type: RequestType,
    op_name: &str,
) -> anyhow::Result<Vec<tokio::sync::oneshot::Receiver<()>>> {
    assert_eq!(
        block_pairs_per_group.len(),
        num_groups,
        "{op_name}: one block-pair list per group required (got {} for {num_groups} groups)",
        block_pairs_per_group.len()
    );
    let mut notify_receivers = Vec::with_capacity(num_groups);
    for (group, group_block_pairs) in block_pairs_per_group.iter().enumerate() {
        let block_xfer_req = BlockTransferRequest {
            from_pool,
            to_pool,
            blocks: group_block_pairs.clone(),
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

    // The tier SOURCE ids are shared across groups (one tier pool); each
    // group writes into its OWN destination device ids. Legacy single-group
    // onboards carry no per-group mapping and use group 0's destinations
    // for every group, exactly as before. With the per-group onboard
    // geometry (fix 2), a group's row carries its OWN ext-block count: a
    // coarse-granularity group takes FEWER blocks than the flat
    // scheduler-granularity count, and its leg's pairing truncates to its
    // own count (the leading tier ids pair with its destinations).
    let dst_ids_per_group: Vec<Vec<usize>> = if onboard_req.per_group_dst_block_ids.is_empty() {
        vec![onboard_req.dst_block_ids.clone(); num_groups]
    } else {
        onboard_req.per_group_dst_block_ids.clone()
    };
    // Phase A: per-group SOURCE offsets into the staged tier-2 list — the
    // mamba groups' leg pairs its checkpoint destination with the LAST
    // staged block instead of the first.
    let src_offsets = &onboard_req.per_group_src_offsets;
    let block_pairs_per_group: Vec<Vec<(usize, usize)>> = dst_ids_per_group
        .into_iter()
        .enumerate()
        .map(|(group, dst_ids)| {
            let src_offset = src_offsets.get(group).copied().unwrap_or(0);
            debug_assert!(
                src_offset + dst_ids.len() <= src_block_ids.len(),
                "group {group}: source offset ({src_offset}) + destinations ({}) exceed \
                 the source blocks ({}) — a group's leg carries its OWN pairing",
                dst_ids.len(),
                src_block_ids.len()
            );
            src_block_ids[src_offset..]
                .iter()
                .copied()
                .zip(dst_ids)
                .collect()
        })
        .collect();

    // TEMPORARY ONBOARD-PROBE (debug level — enable via
    // DYN_LOG=info,_core::block_manager=debug; remove after the H2D content
    // fix): the source tier slots + their first bytes and the first dst
    // device ids — pairs with the STORE-PROBE's allocated ids and the
    // pinned-run's same probe to discriminate slot-map divergence from
    // content divergence.
    tracing::debug!(
        request_id = %request_id,
        operation_id = %operation_id,
        n_src = onboard_req.src_blocks.len(),
        src_probe = ?onboard_req.src_blocks.probe_heads(3),
        dst_first = ?block_pairs_per_group
            .first()
            .map(|g| g.iter().take(3).collect::<Vec<_>>()),
        "ONBOARD-PROBE: onboard issued"
    );

    // Issue the onboard: ONE logical op fanned out to one request PER GROUP
    // (shared tier sources into each group's own device destinations), each
    // leg with its own deterministic uuid. All legs are sent before any is
    // awaited.
    let notify_receivers = fan_out_group_legs(
        leader,
        num_groups,
        request_id,
        *operation_id,
        onboard_req.src_blocks.storage_pool(),
        BlockTransferPool::Device,
        &block_pairs_per_group,
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

    /// TEMPORARY ONBOARD-PROBE support (remove after the H2D content fix):
    /// (block_id, first-16-bytes-hex) of the first `n` blocks — the source
    /// content discriminator between the pinned and devbar tiers.
    fn probe_heads(&self, n: usize) -> Vec<(BlockId, String)>;
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

    fn probe_heads(&self, n: usize) -> Vec<(BlockId, String)> {
        use dynamo_llm::block_manager::block::data::{BlockDataExt as _, BlockDataProvider as _};
        self.blocks
            .iter()
            .take(n)
            .map(|b| {
                // The host tier is FullyContiguous — the whole-block view is
                // the readable surface (layer_view is the LayerSeparate API).
                let head = b
                    .block_data()
                    .block_view()
                    .ok()
                    .map(|v| {
                        let take = 16.min(v.size());
                        // SAFETY: the view covers `v.size()` >= `take`
                        // readable bytes of pool-owned storage that outlives
                        // this probe read.
                        let bytes: Vec<u8> =
                            unsafe { std::slice::from_raw_parts(v.as_ptr(), take) }.to_vec();
                        bytes.iter().map(|x| format!("{x:02x}")).collect::<String>()
                    })
                    .unwrap_or_else(|| "n/a".to_string());
                (b.block_id(), head)
            })
            .collect()
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
        create_test_slot_with_groups_mamba(
            num_tokens,
            offload_min_priority,
            num_groups,
            Vec::new(),
            0,
        )
    }

    /// Variant with the Phase A mamba restore geometry configured (the
    /// slot's mamba groups + their TRUE state-checkpoint cadence in
    /// tokens). NOTE: for the mamba tests the block size is 32 (the test
    /// constant), so pass the mamba block tokens in the same units the
    /// slot will divide by — the ARITHMETIC is what's under test, not the
    /// fp4's constants.
    #[allow(clippy::too_many_arguments)]
    fn create_test_slot_with_groups_mamba(
        num_tokens: usize,
        offload_min_priority: u32,
        num_groups: usize,
        mamba_groups: Vec<usize>,
        mamba_block_tokens: usize,
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
            mamba_groups,
            mamba_block_tokens,
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
        drain_offload_requests(rx)
            .into_iter()
            .map(|offload| offload.block_ids)
            .collect()
    }

    /// Drains all pending offload requests from the channel, whole.
    fn drain_offload_requests(
        rx: &mut mpsc::UnboundedReceiver<LocalTransferRequest>,
    ) -> Vec<LocalOffloadRequest> {
        let mut results = Vec::new();
        while let Ok(req) = rx.try_recv() {
            if let LocalTransferRequest::Offload(offload) = req {
                results.push(offload);
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
    // Test 1b: the 2026-10-24 ladder fatal — vLLM re-queries
    // get_num_new_matched_tokens while a previous onboard is IN FLIGHT
    // (slot state Onboarding). The lookup must skip gracefully (the
    // in-flight match completes on its own; the step answers 0 new
    // external tokens), never fatal the EngineCore.
    // ---------------------------------------------------------------
    #[test]
    fn test_lookup_skips_while_onboarding() {
        let (mut slot, _rx) = create_test_slot(96, 0);
        slot.state = SlotState::Onboarding(14896);
        slot.acquire_local_matches(0)
            .expect("an in-flight onboard must skip the re-lookup, not fatal");
    }

    // ---------------------------------------------------------------
    // Test 1c: the 2026-10-24 log-flood fix — the under-coverage emission
    // is STRUCTURAL for multi-group slots (the flat list is group 0's
    // mamba row vs the attention block math → always under-covers) and
    // must not mark the request warned; single-group slots warn ONCE per
    // request (the COW transient bursts per step).
    // ---------------------------------------------------------------
    #[test]
    fn test_under_cover_warn_rate_limit() {
        // Multi-group (HMA): the flag NEVER sets (the emission is debug).
        let (mut slot, _rx) = create_test_slot_with_groups(96, 0, 4);
        slot.current_position = 90 * 32; // beyond the tracked coverage
        slot.apply_scheduler_output(&[], &[], 96, 96, None, None)
            .unwrap();
        assert!(
            !slot.under_cover_warned,
            "multi-group under-coverage is structural — never the warned path"
        );

        // Single-group: the FIRST under-covering step sets the flag; the
        // second does not re-mark (the emission is once-per-request).
        let (mut slot, _rx) = create_test_slot(96, 0);
        slot.current_position = 90 * 32;
        slot.apply_scheduler_output(&[], &[], 96, 96, None, None)
            .unwrap();
        assert!(
            slot.under_cover_warned,
            "single-group under-coverage warns (the first time)"
        );
        slot.apply_scheduler_output(&[], &[], 96, 96, None, None)
            .unwrap();
        assert!(
            slot.under_cover_warned,
            "the flag stays set (once-per-request)"
        );
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
    // Test 11: mid-list mismatch degrades to replace-with-warn (was a
    //          contract-violation panic). vLLM's HMA hybrid feeds can
    //          carry shapes the ladder cannot align; the feed is the
    //          authoritative current list — [10,11,12] + apply([11,14])
    //          → [11,14].
    // ---------------------------------------------------------------
    #[test]
    fn test_mid_list_mismatch_replaces_with_the_feed() {
        let num_tokens = 128;
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        slot.append_mutable_device_blocks(&[10, 11, 12]).unwrap();

        // block_ids[0]=11 is found at device_blocks[1], but the suffix
        // [11,12] does not match the feed's prefix [11,14] — replace.
        slot.apply_scheduler_output(&[], &[11, 14], 0, 128, None, None)
            .unwrap();

        assert_eq!(slot.device_blocks_snapshot(), &[11, 14]);
    }

    // ---------------------------------------------------------------
    // Test 12: Non-contiguous duplicate extends with a warning (relaxed).
    //          [10,11,12,13,14] + apply([13,14,10]) — the suffix overlap
    //          [13,14] matches; the tail id 10 is already tracked.
    //          Historically a contract-violation panic; relaxed because
    //          vLLM's HMA mamba groups legitimately report duplicate ids
    //          within one request (live evidence: [0,0,0,0,11,18]) — the
    //          alignment cases (1/2/3/b/c) are the real consistency
    //          guards and case (d) still fails loud on genuinely
    //          inconsistent feeds.
    // ---------------------------------------------------------------
    #[test]
    fn test_non_contiguous_duplicate_extends_with_warning() {
        let num_tokens = 192; // 6 blocks
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        slot.append_mutable_device_blocks(&[10, 11, 12, 13, 14])
            .unwrap();

        // Overlap: suffix [13,14] matches prefix [13,14], overlap=2.
        // new_ids = [10] — already tracked: warn, extend (no panic).
        slot.apply_scheduler_output(&[], &[13, 14, 10], 0, 192, None, None)
            .unwrap();

        assert_eq!(slot.device_blocks_snapshot(), &[10, 11, 12, 13, 14, 10]);
    }

    // ---------------------------------------------------------------
    // Test 12b: IDENTICAL duplicate-heavy lists are a no-op (the node
    //           regression): vLLM's mamba group reported
    //           [0,0,0,0,11,18] as BOTH the tracked list and the next
    //           feed; the old rposition search aligned at the LAST 0
    //           (pos 3) and panicked on a false suffix mismatch.
    // ---------------------------------------------------------------
    #[test]
    fn test_reconcile_equal_duplicate_lists_is_a_no_op() {
        let num_tokens = 128;
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        slot.append_mutable_device_blocks(&[0, 0, 0, 0, 11, 18])
            .unwrap();

        // The exact same feed — ladder case 1 short-circuits before any
        // id search; no panic, no duplication.
        slot.apply_scheduler_output(&[], &[0, 0, 0, 0, 11, 18], 0, 128, None, None)
            .unwrap();

        assert_eq!(slot.device_blocks_snapshot(), &[0, 0, 0, 0, 11, 18]);
    }

    // ---------------------------------------------------------------
    // Test 12c: the feed matches the tracked list's natural suffix —
    //           already applied, no-op (ladder case 2).
    // ---------------------------------------------------------------
    #[test]
    fn test_reconcile_natural_suffix_is_a_no_op() {
        let num_tokens = 96;
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        slot.append_mutable_device_blocks(&[5, 6, 7]).unwrap();

        // [6,7] is the tracked list's natural suffix — nothing to do.
        slot.apply_scheduler_output(&[], &[6, 7], 0, 96, None, None)
            .unwrap();

        assert_eq!(slot.device_blocks_snapshot(), &[5, 6, 7]);
    }

    // ---------------------------------------------------------------
    // Test 12d: prefix extend with a duplicated head — the tracked list
    //           is a prefix of the feed, and the feed's head repeats.
    //           Ladder case 3 must align at position 0, not the last
    //           occurrence of the head (the old rposition trap).
    // ---------------------------------------------------------------
    #[test]
    fn test_reconcile_prefix_extend_with_duplicate_head() {
        let num_tokens = 96;
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        slot.append_mutable_device_blocks(&[0, 0]).unwrap();

        // feed [0,0,9] extends the tracked [0,0] — the old code's
        // rposition would find head 0 at pos 1 (the LAST 0) and
        // miscompare; the prefix case aligns correctly at 0.
        slot.apply_scheduler_output(&[], &[0, 0, 9], 0, 96, None, None)
            .unwrap();

        assert_eq!(slot.device_blocks_snapshot(), &[0, 0, 9]);
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
    // Test 15: COW block reassignment — head matches, tail diverged.
    //          vLLM's kv_cache_manager replaced a prefix-cache-shared
    //          block with a private copy (req_blocks[i] = cow_block),
    //          so the feed is the request's authoritative CURRENT list:
    //          [0,4] + apply([0,6]) → [0,6] (replace, no panic).
    // ---------------------------------------------------------------
    #[test]
    fn test_cow_head_match_replaces_divergent_tail() {
        let num_tokens = 64; // 2 blocks at BLOCK_SIZE 32
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        // Admission published the request's full list via
        // update_state_after_alloc; block 4 was shared with the prefix cache.
        slot.append_mutable_device_blocks(&[0, 4]).unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 2);

        // The request wrote into block 4's position; vLLM reassigned it to
        // the private COW copy 6 and re-published the full list.
        slot.apply_scheduler_output(&[], &[0, 6], 0, num_tokens, None, None)
            .unwrap();

        assert_eq!(slot.num_device_blocks_allocated(), 2);
        assert_eq!(slot.device_blocks_snapshot(), &[0, 6]);
    }

    // ---------------------------------------------------------------
    // Test 16: COW reassignment with a longer feed — the replacement
    //          carries new blocks too: [0,4] + [0,6,7] → [0,6,7].
    // ---------------------------------------------------------------
    #[test]
    fn test_cow_head_match_replacement_with_new_tail_blocks() {
        let num_tokens = 96; // 3 blocks
        let (mut slot, _rx) = create_test_slot(num_tokens, 0);

        slot.append_mutable_device_blocks(&[0, 4]).unwrap();

        slot.apply_scheduler_output(&[], &[0, 6, 7], 0, num_tokens, None, None)
            .unwrap();

        assert_eq!(slot.num_device_blocks_allocated(), 3);
        assert_eq!(slot.device_blocks_snapshot(), &[0, 6, 7]);
    }

    // ---------------------------------------------------------------
    // Test 17: COW reassignment reconciles the per-group table the same
    //          way — the replaced positions update in EVERY group's row
    //          while rows without reassignment keep extending.
    // ---------------------------------------------------------------
    #[test]
    fn test_cow_reconciles_per_group_block_table() {
        let num_tokens = 96;
        let (mut slot, _rx) = create_test_slot_with_groups(num_tokens, 0, 4);

        // The HMA-PROBE shape from the node: the long request's full
        // per-group lists at update_state_after_alloc.
        slot.set_per_group_device_blocks(&[vec![0, 4], vec![0, 3], vec![0, 2], vec![1, 5]])
            .unwrap();

        // The COW step re-published full per-group lists with group 0's
        // second block reassigned 4 → 6.
        slot.append_per_group_device_blocks(&[vec![0, 6], vec![0, 3], vec![0, 2], vec![1, 5]])
            .unwrap();

        assert_eq!(
            slot.per_group_block_table_snapshot(),
            Some(&[vec![0, 6], vec![0, 3], vec![0, 2], vec![1, 5]][..])
        );

        // A subsequent genuine delta still appends at the tail.
        slot.append_per_group_device_blocks(&[vec![9], vec![9], vec![9], vec![9]])
            .unwrap();
        assert_eq!(
            slot.per_group_block_table_snapshot(),
            Some(&[vec![0, 6, 9], vec![0, 3, 9], vec![0, 2, 9], vec![1, 5, 9]][..])
        );
    }

    // ---------------------------------------------------------------
    // Test 18: the 2026-10-21 fatal shape — an onboard geometry whose
    // absolute computed-position slice indexes past a checkpoint-
    // indexed HMA group row (mamba rows are entry-indexed, shorter
    // than the computed token range; the live worker died on "needs
    // positions 10..19, the group has 10"). The destinations must fall
    // back to the row's tail; a token-position-indexed row that
    // covers the range keeps the exact absolute slice.
    // ---------------------------------------------------------------
    #[test]
    fn test_onboard_geometry_tail_fallback_for_short_group_rows() {
        let num_tokens = 32 * 64; // ample blocks for the shape below
        let (mut slot, _rx) = create_test_slot_with_groups(num_tokens, 0, 4);

        // Group 3 (attention, token-position-indexed): the full 19-block
        // computed range. Groups 0..2 (mamba, checkpoint-indexed): 10
        // entries for the same token range — the live [4,4,4,12] shape.
        let attention_row = block_ids(100, 19);
        let mut rows: Vec<Vec<BlockId>> =
            vec![block_ids(200, 10), block_ids(300, 10), block_ids(400, 10)];
        rows.push(attention_row.clone());
        slot.set_per_group_device_blocks(&rows).unwrap();

        // The death's geometry: 7,840 internal + 7,056 external tokens,
        // every group at bspb 784 → start=10, count=9 per group.
        let geometry = OnboardGeometry {
            starts: vec![10, 10, 10, 10],
            ext_counts: vec![9, 9, 9, 9],
        };

        let dst_rows = slot
            .onboard_geometry_dst_rows(&geometry)
            .expect("a short checkpoint-indexed row must tail-fallback, not fatal");

        // The attention row covers the absolute range: exact slice [10..19).
        assert_eq!(dst_rows[3], attention_row[10..19]);
        // The mamba rows cannot host positions 10..19 (len 10): the
        // destinations are their TAIL 9 entries (the allocator appends the
        // external destinations at the row's end).
        assert_eq!(dst_rows[0], rows[0][1..10]);
        assert_eq!(dst_rows[1], rows[1][1..10]);
        assert_eq!(dst_rows[2], rows[2][1..10]);
    }

    // ---------------------------------------------------------------
    // Test 19: a row too short to host even `count` destinations is a
    // genuine contract violation — the loud error must survive the
    // tail-fallback fix.
    // ---------------------------------------------------------------
    #[test]
    fn test_onboard_geometry_uncoverable_row_still_errors() {
        let (mut slot, _rx) = create_test_slot_with_groups(32 * 64, 0, 2);
        slot.set_per_group_device_blocks(&[block_ids(100, 2), block_ids(200, 2)])
            .unwrap();
        let geometry = OnboardGeometry {
            starts: vec![0, 0],
            ext_counts: vec![3, 3],
        };
        assert!(
            slot.onboard_geometry_dst_rows(&geometry).is_err(),
            "a row shorter than the ext count has nothing to host — must stay loud"
        );
    }

    // ---------------------------------------------------------------
    // Test 20: the Phase A chain-end registry — a multi-group external
    // match is valid ONLY at a recorded chain end (the tier-2 content
    // there carries the mamba checkpoint AT the boundary).
    // ---------------------------------------------------------------
    #[test]
    fn test_hybrid_chain_end_registry() {
        let h: SequenceHash = 0xDEAD_BEEF;
        assert!(
            !is_hybrid_chain_end(h),
            "an unrecorded hash must not validate a multi-group match"
        );
        record_hybrid_chain_end(h);
        assert!(
            is_hybrid_chain_end(h),
            "a recorded chain end must validate the match ending there"
        );
        assert!(
            !is_hybrid_chain_end(0xBEEF_CAFE),
            "other hashes stay refused"
        );
    }

    // ---------------------------------------------------------------
    // Test 21: the Phase A mamba restore plan — the engine's state-slot
    // arithmetic ((P-1)/mamba_block_tokens) against the 2026-10-21 fatal
    // shape: P=14,896 tokens with mamba rows of 10 entries → slot 7
    // (scaled to the test block size: the ARITHMETIC is under test).
    // ---------------------------------------------------------------
    #[test]
    fn test_mamba_restore_plan_targets_engine_state_slot() {
        // The 2026-10-21 fatal, scaled: current_position 7,840 tokens,
        // num_external 7,056 → P=14,896; mamba cadence 1,960 → slot 7.
        // The test block size is 32, so build the equivalent in 32-token
        // units: use mamba_block_tokens=1960 directly with the REAL numbers
        // (the slot divides tokens, not blocks).
        let (mut slot, _rx) =
            create_test_slot_with_groups_mamba(32 * 1024, 0, 4, vec![0, 1, 2], 1960);
        slot.current_position = 7840;
        // staged (contents unused by the plan builder)
        slot.staging_from_host = Some(Vec::new());
        // The finish-shape rows: mamba rows long enough for slot 7.
        slot.set_per_group_device_blocks(&[
            block_ids(200, 10),
            block_ids(300, 10),
            block_ids(400, 10),
            block_ids(100, 19),
        ])
        .unwrap();

        let plan = slot
            .build_mamba_restore_plan(7056)
            .expect("the plan must build for a covered row")
            .expect("the mamba contract is configured and staging exists");
        assert_eq!(plan.dst_per_group.len(), 3, "one slot per mamba group");
        assert_eq!(plan.dst_per_group[&0], 207, "group 0 row[7]");
        assert_eq!(plan.dst_per_group[&1], 307, "group 1 row[7]");
        assert_eq!(plan.dst_per_group[&2], 407, "group 2 row[7]");
        assert!(
            !plan.dst_per_group.contains_key(&3),
            "the attention group is not in the mamba plan"
        );

        // The uncovered-row case stays LOUD: refusing wrong bytes.
        slot.set_per_group_device_blocks(&[
            block_ids(200, 4), // shorter than slot 7
            block_ids(300, 10),
            block_ids(400, 10),
            block_ids(100, 19),
        ])
        .unwrap();
        assert!(
            slot.build_mamba_restore_plan(7056).is_err(),
            "an uncovered state slot must error, not restore wrong bytes"
        );
    }

    // ---------------------------------------------------------------
    // Test 22: the Phase A onboard leg shape — the mamba groups' leg
    // carries ONE destination (the engine state slot) with a source
    // offset pointing at the LAST staged block; the attention group
    // keeps the positional pairing.
    // ---------------------------------------------------------------
    #[test]
    fn test_onboard_blocks_mamba_leg_shape() {
        struct FakeSrc(Vec<BlockId>);
        impl AnyBlocks for FakeSrc {
            fn len(&self) -> usize {
                self.0.len()
            }
            fn storage_pool(&self) -> BlockTransferPool {
                BlockTransferPool::Host
            }
            fn block_ids(&self) -> Vec<BlockId> {
                self.0.clone()
            }
            fn probe_heads(&self, n: usize) -> Vec<(BlockId, String)> {
                self.0.iter().take(n).map(|&id| (id, "x".into())).collect()
            }
        }

        let (mut slot, mut rx) =
            create_test_slot_with_groups_mamba(32 * 64, 0, 4, vec![0, 1, 2], 1960);
        slot.evaluated_blocks = 0;
        slot.set_per_group_device_blocks(&[
            block_ids(200, 10),
            block_ids(300, 10),
            block_ids(400, 10),
            block_ids(100, 5),
        ])
        .unwrap();
        let geometry = OnboardGeometry {
            starts: vec![0, 0, 0, 0],
            ext_counts: vec![5, 5, 5, 5],
        };
        let plan = MambaRestorePlan {
            dst_per_group: HashMap::from([(0usize, 207usize), (1, 307), (2, 407)]),
        };
        let src = FakeSrc(block_ids(1000, 5));
        slot.onboard_blocks(
            Box::new(src),
            block_ids(500, 5),
            Some(&geometry),
            Some(&plan),
        )
        .unwrap();

        let req = rx.try_recv().expect("the onboard request must be queued");
        let LocalTransferRequest::Onboard(onboard) = req else {
            panic!("expected an onboard request");
        };
        // The attention group keeps the positional geometry slice of its
        // OWN table row (starts[3]=0, ext[3]=5 → row 3's first 5 ids).
        assert_eq!(onboard.per_group_dst_block_ids[3], block_ids(100, 5));
        assert_eq!(onboard.per_group_src_offsets[3], 0);
        // Each mamba group: ONE destination (the state slot) sourced from
        // the LAST staged block (offset 5-1=4).
        for (group, slot_id) in [(0, 207usize), (1, 307), (2, 407)] {
            assert_eq!(
                onboard.per_group_dst_block_ids[group],
                vec![slot_id],
                "mamba group {group} carries exactly the state slot"
            );
            assert_eq!(
                onboard.per_group_src_offsets[group], 4,
                "mamba group {group} pairs with the last staged block"
            );
        }
    }

    // ---------------------------------------------------------------
    // Test 18: The cm008 chunked-prefill COW cadence — a non-resumed
    //          step's replacement-shaped feed ([0,6]) under-covers the
    //          tokens scheduled so far. Must not panic (the former
    //          `next_position` assert killed EngineCore), must not
    //          produce candidates beyond the tracked list, and the next
    //          full-feed step must restore growth and resume offloads
    //          for real entries only — the reassigned-away id (1) is
    //          never offloaded.
    // ---------------------------------------------------------------
    #[test]
    fn test_cow_step_under_coverage_clamps_and_self_heals() {
        let num_tokens = 192; // 6 blocks at BLOCK_SIZE 32
        let (mut slot, mut rx) = create_test_slot(num_tokens, 0);

        // Chunk 1 (new request): full feed [0,1], 64 tokens scheduled —
        // both blocks become offload candidates.
        slot.apply_scheduler_output(&[], &[0, 1], 0, 64, None, None)
            .unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 2);
        assert_eq!(drain_offload_block_ids(&mut rx), vec![vec![0, 1]]);

        // Chunk 2 (the COW step): vLLM reassigned the shared tail (1 → 6)
        // and fed the replacement-shaped list while the cumulative
        // scheduled tokens (128) outgrew the tracked list (2 × 32). The
        // former assert panicked here.
        slot.apply_scheduler_output(&[], &[0, 6], 64, 32, None, None)
            .unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 2); // replaced, not doubled
        assert_eq!(slot.device_blocks_snapshot(), &[0, 6]);
        assert!(
            drain_offload_block_ids(&mut rx).is_empty(),
            "no candidates beyond the tracked list on the under-covered step"
        );

        // Chunk 3: the full feed restores coverage; offloads resume for the
        // newly covered real entry only.
        slot.apply_scheduler_output(&[], &[0, 6, 7, 8], 96, 32, None, None)
            .unwrap();
        assert_eq!(slot.num_device_blocks_allocated(), 4);
        assert_eq!(slot.device_blocks_snapshot(), &[0, 6, 7, 8]);
        assert_eq!(drain_offload_block_ids(&mut rx), vec![vec![8]]);
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

    // ---------------------------------------------------------------
    // Hybrid mid-flight suppression (fix 1a): multi-group slots create
    // NO offload op mid-flight — the fp4 COWs its chunk-boundary block
    // on every chunked-prefill boundary ([0,1]→[0,6]→[0,11]→[0,16]
    // live evidence), so a mid-flight copy captures pre-write content
    // of a soon-reassigned id, and BOTH the stale and a later final
    // copy would match the same hash. The candidates stay for the
    // finish flush; evaluated_blocks is frozen. Single-group keeps
    // today's op (pinned by test_offload_sends_correct_block_ids).
    // ---------------------------------------------------------------
    #[test]
    fn hybrid_mid_flight_offload_is_suppressed_and_candidates_frozen() {
        let num_tokens = 96; // 3 blocks
        let (mut slot, mut rx) = create_test_slot_with_groups(num_tokens, 0, 3);

        // The mapping is published as usual — the suppression must hold
        // even with everything a mid-flight offload would need.
        slot.set_per_group_device_blocks(&[
            block_ids(100, 3),
            block_ids(200, 3),
            block_ids(300, 3),
        ])
        .unwrap();

        slot.append_mutable_device_blocks(&block_ids(100, 3))
            .unwrap();
        slot.apply_scheduler_output(&[], &[], 0, num_tokens, None, None)
            .unwrap();

        // No transfer request, no metadata op, evaluated_blocks frozen.
        assert!(
            drain_offload_requests(&mut rx).is_empty(),
            "hybrid mid-flight offload must be suppressed"
        );
        assert!(
            slot.take_pending_operations().is_none(),
            "hybrid mid-flight offload must record no metadata operations"
        );
        assert_eq!(
            slot.evaluated_blocks, 0,
            "the candidates must stay for the finish flush"
        );
        assert_eq!(slot.current_position, num_tokens);

        // Single-group keeps today's mid-flight op (regression pin).
        let (mut slot, mut rx) = create_test_slot(num_tokens, 0);
        slot.append_mutable_device_blocks(&block_ids(100, 3))
            .unwrap();
        slot.apply_scheduler_output(&[], &[], 0, num_tokens, None, None)
            .unwrap();
        assert_eq!(
            drain_offload_block_ids(&mut rx),
            vec![vec![100, 101, 102]],
            "attention-only must keep the validated mid-flight path"
        );
    }

    // ---------------------------------------------------------------
    // The hybrid finish flush (fix 1b): after mark_as_finished, the
    // remaining completed positions are offloaded using the CURRENT
    // (final) per-group table — the Python finish-publish refreshed it
    // BEFORE the flush, so the COW-reassigned ids are what move. The
    // transfer request is sent immediately; the per-leg metadata ops
    // are RETURNED for the leader to defer into the next
    // build_connector_metadata. evaluated_blocks advances; a second
    // call is None; single-group returns None.
    // ---------------------------------------------------------------
    #[test]
    fn hybrid_finish_flush_offloads_final_table_ids_and_defers_leg_ops() {
        let num_tokens = 96; // 3 blocks at BLOCK_SIZE 32
        let (mut slot, mut rx) = create_test_slot_with_groups(num_tokens, 0, 2);

        // Mid-flight suppression left every completed position for the
        // flush; the admission publish set the initial mapping.
        slot.set_per_group_device_blocks(&[block_ids(100, 3), block_ids(900, 3)])
            .unwrap();
        slot.append_mutable_device_blocks(&block_ids(100, 3))
            .unwrap();
        slot.apply_scheduler_output(&[], &[], 0, num_tokens, None, None)
            .unwrap();
        assert!(drain_offload_requests(&mut rx).is_empty());

        // The finish publish REPLACES the table with the FINAL ids: the
        // chunk-boundary COW reassigned group 0's position 1 (101 → 6).
        slot.set_per_group_device_blocks(&[vec![100, 6, 102], vec![900, 901, 902]])
            .unwrap();

        slot.mark_as_finished(1).unwrap();

        let ops = slot
            .take_hybrid_flush_offload()
            .expect("the flush must not error")
            .expect("a multi-group slot with 3 completed blocks must flush");
        assert_eq!(ops.len(), 2, "one metadata op per group leg");
        for (group, op) in ops.iter().enumerate() {
            assert_eq!(op.request_id, "test-req");
            assert_eq!(op.transfer_type, TransferType::Store);
            assert_eq!(op.request_type, RequestType::Flush);
            assert_eq!(
                op.uuid,
                derive_leg_uuid(ops[0].uuid, group),
                "leg {group} uuid must be the deterministic derivative of leg 0"
            );
        }
        assert_ne!(ops[0].uuid, ops[1].uuid);
        assert_eq!(ops[0].uuid, derive_leg_uuid(ops[0].uuid, 0));

        // The transfer request carries the FINAL table ids — the stale,
        // reassigned-away group-0 id (101) is never offloaded.
        let offloads = drain_offload_requests(&mut rx);
        assert_eq!(offloads.len(), 1);
        assert_eq!(offloads[0].block_ids, vec![100, 6, 102]);
        assert_eq!(
            offloads[0].per_group_block_ids,
            vec![vec![100, 6, 102], vec![900, 901, 902]]
        );
        assert_eq!(offloads[0].priorities, vec![0, 0, 0]);

        // The flush consumed the candidates: evaluated advanced, a
        // second call has nothing left.
        assert_eq!(slot.evaluated_blocks, 3);
        assert!(slot.take_hybrid_flush_offload().unwrap().is_none());
    }

    #[test]
    fn hybrid_finish_flush_returns_none_for_single_group_and_zero_count() {
        // Attention-only: byte-identical mid-flight path, never a flush.
        let num_tokens = 96;
        let (mut slot, mut rx) = create_test_slot(num_tokens, 0);
        slot.append_mutable_device_blocks(&block_ids(100, 3))
            .unwrap();
        slot.apply_scheduler_output(&[], &[], 0, num_tokens, None, None)
            .unwrap();
        assert_eq!(drain_offload_requests(&mut rx).len(), 1, "today's op");
        slot.mark_as_finished(1).unwrap();
        assert!(slot.take_hybrid_flush_offload().unwrap().is_none());

        // Multi-group with nothing completed: count == 0 → None.
        let (mut slot, _rx) = create_test_slot_with_groups(96, 0, 2);
        slot.set_per_group_device_blocks(&[block_ids(100, 3), block_ids(900, 3)])
            .unwrap();
        slot.mark_as_finished(1).unwrap();
        assert!(slot.take_hybrid_flush_offload().unwrap().is_none());
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

        fn probe_heads(&self, _n: usize) -> Vec<(BlockId, String)> {
            Vec::new()
        }
    }

    #[test]
    fn onboard_records_one_metadata_operation_per_group() {
        let (mut slot, _rx) = create_test_slot_with_groups(96, 0, 2);

        // Multi-group onboards resolve destination ids per group from the
        // mapping — publish it first (positions 0..2 hold the onboard dst).
        slot.set_per_group_device_blocks(&[vec![0, 1], vec![90, 91]])
            .unwrap();

        slot.onboard_blocks(
            Box::new(TestAnyBlocks {
                ids: vec![500, 501],
                pool: BlockTransferPool::Host,
            }),
            vec![0, 1],
            None,
            None,
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

    // ---------------------------------------------------------------
    // Per-group block-id mapping (HMA): set/append + consumers
    // ---------------------------------------------------------------

    /// Drains all pending onboard requests from the channel, whole.
    fn drain_onboard_requests(
        rx: &mut mpsc::UnboundedReceiver<LocalTransferRequest>,
    ) -> Vec<LocalOnboardRequest> {
        let mut results = Vec::new();
        while let Ok(req) = rx.try_recv() {
            if let LocalTransferRequest::Onboard(onboard) = req {
                results.push(onboard);
            }
        }
        results
    }

    #[test]
    fn per_group_mapping_set_replaces_and_append_extends() {
        let (mut slot, _rx) = create_test_slot_with_groups(96, 0, 2);

        // update_state_after_alloc / new-request scheduling report FULL
        // per-group lists — they REPLACE the table.
        slot.set_per_group_device_blocks(&[vec![10, 11], vec![20, 22]])
            .unwrap();
        assert_eq!(
            slot.per_group_block_table_snapshot(),
            Some(&[vec![10, 11], vec![20, 22]][..])
        );

        // Running-request scheduling reports per-group DELTAS — they extend
        // each group's list at the tail.
        slot.append_per_group_device_blocks(&[vec![12], vec![23]])
            .unwrap();
        assert_eq!(
            slot.per_group_block_table_snapshot(),
            Some(&[vec![10, 11, 12], vec![20, 22, 23]][..])
        );

        // A second full report replaces everything again (e.g. resume after
        // preemption re-publishes the request's full lists).
        slot.set_per_group_device_blocks(&[vec![30], vec![40]])
            .unwrap();
        assert_eq!(
            slot.per_group_block_table_snapshot(),
            Some(&[vec![30], vec![40]][..])
        );
    }

    #[test]
    fn per_group_mapping_delta_append_before_any_set_builds_table() {
        let (mut slot, _rx) = create_test_slot_with_groups(64, 0, 2);
        slot.append_per_group_device_blocks(&[vec![5], vec![6]])
            .unwrap();
        assert_eq!(
            slot.per_group_block_table_snapshot(),
            Some(&[vec![5], vec![6]][..])
        );
    }

    #[test]
    fn per_group_mapping_wrong_group_count_errors() {
        let (mut slot, _rx) = create_test_slot_with_groups(64, 0, 2);
        let err = slot
            .set_per_group_device_blocks(&[vec![1, 2]])
            .expect_err("1 list for a 2-group slot must fail");
        assert!(err.to_string().contains("expects 2 group lists"));
        let err = slot
            .append_per_group_device_blocks(&[vec![1], vec![2], vec![3]])
            .expect_err("3 lists for a 2-group slot must fail");
        assert!(err.to_string().contains("expects 2 group lists"));
    }

    #[test]
    fn offload_single_group_keeps_legacy_empty_per_group_ids() {
        let num_tokens = 96;
        let (mut slot, mut rx) = create_test_slot_with_groups(num_tokens, 0, 1);

        slot.append_mutable_device_blocks(&block_ids(100, 3))
            .unwrap();
        slot.apply_scheduler_output(&[], &[], 0, num_tokens, None, None)
            .unwrap();

        let offloads = drain_offload_requests(&mut rx);
        assert_eq!(offloads.len(), 1);
        assert!(
            offloads[0].per_group_block_ids.is_empty(),
            "single-group offloads carry no per-group mapping"
        );
    }

    // The mid-flight multi-group offload this test used to pin is
    // suppressed (fix 1a); the refusal now lives on the finish flush:
    // a multi-group slot whose mapping was never published must refuse
    // to move group 0's ids through the other groups' pools there.
    #[test]
    #[should_panic(expected = "per-group device block mapping missing")]
    fn hybrid_finish_flush_without_mapping_refuses_group_zero_ids() {
        let num_tokens = 96;
        let (mut slot, _rx) = create_test_slot_with_groups(num_tokens, 0, 2);

        // Suppressed mid-flight: no panic at apply time even without a
        // mapping (no consumer runs).
        slot.append_mutable_device_blocks(&block_ids(100, 3))
            .unwrap();
        slot.apply_scheduler_output(&[], &[], 0, num_tokens, None, None)
            .unwrap();

        // The finish flush, however, still needs every group's OWN ids.
        slot.mark_as_finished(1).unwrap();
        let _ = slot.take_hybrid_flush_offload().unwrap();
    }

    #[test]
    fn onboard_carries_per_group_destination_ids() {
        let (mut slot, mut rx) = create_test_slot_with_groups(96, 0, 2);

        // Onboard destinations are the mapping's positions 0..2 (the slot's
        // evaluated_blocks is 0 before any onboarding).
        slot.set_per_group_device_blocks(&[vec![7, 8, 9], vec![70, 80, 90]])
            .unwrap();

        slot.onboard_blocks(
            Box::new(TestAnyBlocks {
                ids: vec![500, 501],
                pool: BlockTransferPool::Host,
            }),
            vec![7, 8],
            None,
            None,
        )
        .unwrap();

        let onboards = drain_onboard_requests(&mut rx);
        assert_eq!(onboards.len(), 1);
        let onboard = &onboards[0];
        assert_eq!(onboard.dst_block_ids, vec![7, 8]);
        assert_eq!(
            onboard.per_group_dst_block_ids,
            vec![vec![7, 8], vec![70, 80]]
        );
    }

    // ---------------------------------------------------------------
    // The per-group onboard tail geometry (fix 2): with the published
    // geometry, group g's destinations are its OWN row's slice
    // [starts[g] .. starts[g] + ext_counts[g]] — the vLLM reference's
    // arithmetic (a coarse group's external tail is NOT the flat
    // evaluated-blocks position). Group 0's flat dst list keeps
    // today's semantics. Without geometry, exactly today's positional
    // rows (onboard_carries_per_group_destination_ids pins that
    // fallback).
    // ---------------------------------------------------------------
    #[test]
    fn onboard_geometry_slices_each_group_from_its_own_start() {
        let (mut slot, mut rx) = create_test_slot_with_groups(96, 0, 2);

        slot.set_per_group_device_blocks(&[vec![10, 11, 12, 13, 14], vec![70, 71, 72, 73, 74]])
            .unwrap();
        slot.set_onboard_geometry(&[2, 1], &[3, 2]).unwrap();
        let geometry = slot.onboard_geometry.take();

        // Group 0's flat dst list keeps today's positional semantics:
        // the staged blocks land at the device_blocks positions.
        slot.append_mutable_device_blocks(&[10, 11, 12, 13, 14])
            .unwrap();
        slot.onboard_blocks(
            Box::new(TestAnyBlocks {
                ids: vec![500, 501, 502],
                pool: BlockTransferPool::Host,
            }),
            vec![12, 13, 14],
            geometry.as_ref(),
            None,
        )
        .unwrap();

        let onboards = drain_onboard_requests(&mut rx);
        assert_eq!(onboards.len(), 1);
        assert_eq!(onboards[0].dst_block_ids, vec![12, 13, 14]);
        assert_eq!(
            onboards[0].per_group_dst_block_ids,
            vec![vec![12, 13, 14], vec![71, 72]],
            "group 1's destinations must come from ITS OWN tail [1..3], \
             not the flat position [2..5]"
        );
    }

    #[test]
    fn onboard_geometry_row_too_short_fails_loudly() {
        let (mut slot, _rx) = create_test_slot_with_groups(96, 0, 2);

        slot.set_per_group_device_blocks(&[vec![10, 11], vec![70, 71]])
            .unwrap();
        slot.set_onboard_geometry(&[2, 1], &[3, 2]).unwrap();
        let geometry = slot.onboard_geometry.take();

        let err = slot
            .onboard_blocks(
                Box::new(TestAnyBlocks {
                    ids: vec![500, 501, 502],
                    pool: BlockTransferPool::Host,
                }),
                vec![12, 13, 14],
                geometry.as_ref(),
                None,
            )
            .expect_err("a geometry slice beyond the mapping must fail");
        assert!(
            err.to_string()
                .contains("per-group device block mapping too short"),
            "got: {err}"
        );
    }

    #[test]
    fn set_onboard_geometry_validates_group_count() {
        let (mut slot, _rx) = create_test_slot_with_groups(96, 0, 2);
        let err = slot
            .set_onboard_geometry(&[2], &[3])
            .expect_err("1 entry for a 2-group slot must fail");
        assert!(err.to_string().contains("expects 2 per-group entries"));
    }

    #[test]
    fn onboard_multi_group_without_mapping_refuses_group_zero_destinations() {
        let (mut slot, _rx) = create_test_slot_with_groups(96, 0, 2);
        let err = slot
            .onboard_blocks(
                Box::new(TestAnyBlocks {
                    ids: vec![500, 501],
                    pool: BlockTransferPool::Host,
                }),
                vec![7, 8],
                None,
                None,
            )
            .expect_err("multi-group onboard without a mapping must fail");
        assert!(
            err.to_string()
                .contains("per-group device block mapping missing")
        );
    }
}

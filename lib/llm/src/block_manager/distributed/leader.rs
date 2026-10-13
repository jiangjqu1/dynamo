// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::zmq::*;
use utils::*;

use derive_builder::Builder;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Notify;
use tokio::sync::OnceCell;
use tokio::sync::oneshot;
use tokio::time::sleep;

#[derive(Builder, Clone, Debug, Default)]
pub struct KvbmLeaderNumBlocksConfig {
    #[builder(default = "0.0")]
    pub cache_size_in_gb: f64,

    #[builder(default = "0")]
    pub num_blocks_overriden: usize,
}

fn compute_num_blocks(
    num_blocks_config: &KvbmLeaderNumBlocksConfig,
    bytes_per_block: usize,
) -> usize {
    if num_blocks_config.num_blocks_overriden > 0 {
        num_blocks_config.num_blocks_overriden
    } else {
        ((num_blocks_config.cache_size_in_gb * 1_000_000_000.0) / bytes_per_block as f64) as usize
    }
}

/// Handshake sizing: element-wise Σ across TP workers of each group's
/// `bytes_per_block` (multi-group analog of the legacy scalar Σ), then one
/// shared block count per tier from the Σ total — one host/disk "block
/// unit" spans all groups.
///
/// Tier sizing is deliberately UNBOUNDED by the device count: offload
/// targets come from the tier pool's free-list allocation
/// (`offload.rs` `target_pool.allocate_blocks`), and onboarding matches
/// cached blocks by sequence hash onto arbitrary newly-assigned device
/// blocks — there is no positional device↔host index mapping, so a tier
/// larger than the device pool is a normal, validated configuration
/// (e.g. a 24 GiB host tier behind a ~10 GiB device KV pool).
///
/// Compatibility: workers < hybrid-multigroup send no per-group values
/// (empty vecs) — the scalar `bytes_per_block` Σ is authoritative there,
/// which is also what a mixed legacy/modern TP set degrades to (any legacy
/// peer empties the aggregate; every version emits the scalar). This
/// scalar fallback is reachable on the ZMQ path only for a modern leader
/// decoding legacy payloads via a future zmq.rs shim — see the KNOWN GAP
/// note on `WorkerMetadata::bytes_per_block_per_group` (bincode is
/// positional, so `#[serde(default)]` alone cannot decode a 2-field
/// payload). TODO(hybrid-multigroup): fix the zmq.rs legacy decode and
/// remove the scalar fallback when N-2 no longer includes multigroup-less
/// workers.
fn compute_leader_block_counts(
    host_cfg: &KvbmLeaderNumBlocksConfig,
    disk_cfg: &KvbmLeaderNumBlocksConfig,
    workers: &[WorkerMetadata],
) -> LeaderMetadata {
    let per_group = aggregate_worker_bytes_per_group(
        workers
            .iter()
            .map(|worker| worker.bytes_per_block_per_group.as_slice()),
    );

    let bytes_per_block: usize = if per_group.is_empty() {
        workers.iter().map(|worker| worker.bytes_per_block).sum()
    } else {
        per_group.iter().sum()
    };

    LeaderMetadata {
        num_host_blocks: compute_num_blocks(host_cfg, bytes_per_block),
        num_disk_blocks: compute_num_blocks(disk_cfg, bytes_per_block),
    }
}

#[derive(Builder, Clone, Debug)]
pub struct KvbmLeaderConfig {
    /// The world size.
    #[builder(default = "1")]
    world_size: usize,

    /// The leader-worker init connection timeout seconds.
    #[builder(default = "120")]
    leader_init_timeout_secs: u64,

    #[builder(default = "KvbmLeaderNumBlocksConfig::default()")]
    host_blocks_config: KvbmLeaderNumBlocksConfig,

    #[builder(default = "KvbmLeaderNumBlocksConfig::default()")]
    disk_blocks_config: KvbmLeaderNumBlocksConfig,

    #[builder(default = "String::from(\"tcp://127.0.0.1:56001\")")]
    leader_pub_url: String,

    #[builder(default = "String::from(\"tcp://127.0.0.1:56002\")")]
    leader_ack_url: String,
}

impl KvbmLeaderConfig {
    pub fn builder() -> KvbmLeaderConfigBuilder {
        KvbmLeaderConfigBuilder::default()
    }

    pub fn sanity_check(&self) -> anyhow::Result<()> {
        if self.leader_pub_url == self.leader_ack_url {
            anyhow::bail!(
                "leader_pub_url and leader_ack_url must differ (same endpoint would fail to bind)."
            );
        }

        let cpu = &self.host_blocks_config;
        let disk = &self.disk_blocks_config;
        let cpu_configured = cpu.num_blocks_overriden > 0 || cpu.cache_size_in_gb > 0.0;
        let disk_configured = disk.num_blocks_overriden > 0 || disk.cache_size_in_gb > 0.0;
        if !cpu_configured && !disk_configured {
            panic!(
                "KVBM Configuration Error: At least one cache tier must be configured.\n\
                \n\
                Configure CPU cache (G2) for CPU memory offloading:\n\
                • DYN_KVBM_CPU_CACHE_GB=<size_in_gb>     (e.g., DYN_KVBM_CPU_CACHE_GB=4)\n\
                • DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=<num_blocks>  (e.g., DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=1000)\n\
                \n\
                OR configure disk cache (G3) for direct GPU->Disk offloading:\n\
                • DYN_KVBM_DISK_CACHE_GB=<size_in_gb>     (e.g., DYN_KVBM_DISK_CACHE_GB=8)\n\
                • DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS=<num_blocks>\n\
                \n\
                Note: If only disk cache is configured, KVBM will offload directly from GPU (G1) to Disk (G3), bypassing CPU memory (G2)."
            );
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct KvbmLeaderState {
    pub num_device_blocks: Arc<AtomicUsize>,
    pub num_host_blocks: Arc<AtomicUsize>,
    pub num_disk_blocks: Arc<AtomicUsize>,
    pub workers_allocation_ready: Arc<AtomicBool>,
    pub workers_ready_notify: Arc<Notify>,
}

/// The leader of the KVBM.
///
/// This is responsible for:
/// - Establishing a ZMQ connection with workers.
/// - Syncing the leader barrier with workers.
/// - Sending messages to workers.
pub struct KvbmLeader {
    state: Arc<KvbmLeaderState>,
    zmq_leader: Arc<OnceCell<ZmqActiveMessageLeader>>,
    config: KvbmLeaderConfig,
}

impl KvbmLeader {
    pub async fn new(config: KvbmLeaderConfig) -> anyhow::Result<Self> {
        let leader_sockets = new_leader_sockets(&config.leader_pub_url, &config.leader_ack_url)?;

        let leader = Self {
            state: Arc::new(KvbmLeaderState::default()),
            zmq_leader: Arc::new(tokio::sync::OnceCell::new()),
            config,
        };

        let cancel_token = tokio_util::sync::CancellationToken::new();
        leader.spawn_zmq_task(leader_sockets, cancel_token);

        Ok(leader)
    }

    fn spawn_zmq_task(
        &self,
        leader_sockets: LeaderSockets,
        cancel: tokio_util::sync::CancellationToken,
    ) {
        let cell = self.zmq_leader.clone();
        let state = self.state.clone();
        let world_size = self.config.world_size;
        let timeout = self.config.leader_init_timeout_secs;
        let host_cfg = self.config.host_blocks_config.clone();
        let disk_cfg = self.config.disk_blocks_config.clone();

        // capture num_device_blocks so we can set it inside the closure
        let num_device_blocks_cell = state.num_device_blocks.clone();
        let num_host_blocks_cell = state.num_host_blocks.clone();
        let num_disk_blocks_cell = state.num_disk_blocks.clone();

        tokio::spawn(async move {
            let res = ZmqActiveMessageLeader::new_with_handshake(
                leader_sockets,
                world_size,
                std::time::Duration::from_secs(timeout),
                cancel.clone(),
                move |workers: &[WorkerMetadata]| -> LeaderMetadata {
                    // Record device blocks: min across workers
                    if let Some(min_dev) = workers.iter().map(|w| w.num_device_blocks).min() {
                        num_device_blocks_cell.store(min_dev, Ordering::Release);
                    }

                    let metadata = compute_leader_block_counts(&host_cfg, &disk_cfg, workers);

                    // store into leader state
                    num_host_blocks_cell.store(metadata.num_host_blocks, Ordering::Release);
                    num_disk_blocks_cell.store(metadata.num_disk_blocks, Ordering::Release);

                    metadata
                },
            )
            .await;

            match res {
                Ok(zmq) => {
                    let _ = cell.set(zmq);
                    state
                        .workers_allocation_ready
                        .store(true, Ordering::Release);
                    state.workers_ready_notify.notify_waiters();
                    tracing::info!("ZMQ handshake complete; workers allocation ready");
                }
                Err(e) => {
                    tracing::error!("ZMQ init/handshake failed: {e:?}");
                }
            }
        });
    }

    pub async fn transfer_blocks_request(
        &self,
        request: BlockTransferRequest,
    ) -> anyhow::Result<oneshot::Receiver<()>> {
        let zmq = self
            .zmq_leader
            .get()
            .ok_or_else(|| anyhow::anyhow!("ZMQ leader not ready"))?;
        let data = vec![serde_json::to_vec(&request)?];
        zmq.broadcast(ZMQ_TRANSFER_BLOCKS_MESSAGE, data).await
    }

    pub fn num_device_blocks(&self) -> usize {
        self.state.num_device_blocks.load(Ordering::Acquire)
    }

    pub fn num_host_blocks(&self) -> usize {
        self.state.num_host_blocks.load(Ordering::Acquire)
    }

    pub fn num_disk_blocks(&self) -> usize {
        self.state.num_disk_blocks.load(Ordering::Acquire)
    }

    pub async fn wait_worker_sync_ready(&self) -> bool {
        if self.state.workers_allocation_ready.load(Ordering::Acquire) {
            return true;
        }
        let notified = self.state.workers_ready_notify.notified();
        tokio::select! {
            _ = notified => true,
            _ = sleep(Duration::from_secs(self.config.leader_init_timeout_secs)) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window_cfg(window_bytes: f64) -> KvbmLeaderNumBlocksConfig {
        KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: window_bytes / 1_000_000_000.0,
            num_blocks_overriden: 0,
        }
    }

    fn worker(
        num_device_blocks: usize,
        bytes_per_block: usize,
        groups: Vec<usize>,
    ) -> WorkerMetadata {
        WorkerMetadata {
            num_device_blocks,
            bytes_per_block,
            bytes_per_block_per_group: groups,
        }
    }

    // 24 GiB window in bytes: floor(25,769,803,776 / 89,817,088) = 286.
    const WINDOW_BYTES: f64 = 25_769_803_776.0;
    // attention 12,845,056 + mamba 76,972,032 B/block.
    const GROUP_TOTAL_BYTES_PER_BLOCK: usize = 89_817_088;

    #[test]
    fn per_group_sizing_sums_groups_into_one_shared_block_count() {
        // One modern worker, two groups: the Σ of per-group bytes-per-block
        // sizes ONE shared block count for both tiers.
        let workers = [worker(
            1_000,
            GROUP_TOTAL_BYTES_PER_BLOCK,
            vec![12_845_056, 76_972_032],
        )];
        let meta = compute_leader_block_counts(
            &window_cfg(WINDOW_BYTES),
            &window_cfg(WINDOW_BYTES),
            &workers,
        );
        assert_eq!(meta.num_host_blocks, 286);
        assert_eq!(meta.num_disk_blocks, 286);
    }

    #[test]
    fn tier_sizing_is_not_bounded_by_the_device_block_count() {
        // RESTORED old behavior: a tier may exceed the device pool. Tier
        // pools allocate from free lists and onboarding matches by sequence
        // hash — no positional device↔host mapping — so a 24 GiB host tier
        // behind a small device KV pool is a normal configuration.
        let workers = [worker(
            100,
            GROUP_TOTAL_BYTES_PER_BLOCK,
            vec![12_845_056, 76_972_032],
        )];
        let meta = compute_leader_block_counts(
            &window_cfg(WINDOW_BYTES),
            &window_cfg(WINDOW_BYTES),
            &workers,
        );
        assert_eq!(meta.num_host_blocks, 286);
        assert!(meta.num_host_blocks > 100);
        assert_eq!(meta.num_disk_blocks, 286);
    }

    #[test]
    fn block_override_is_not_capped_by_the_device_count() {
        // DYN_KVBM_*_OVERRIDE_NUM_BLOCKS passes through compute_num_blocks
        // verbatim; it must not be silently clipped to the device count.
        let workers = [worker(100, GROUP_TOTAL_BYTES_PER_BLOCK, vec![])];
        let override_cfg = KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 0.0,
            num_blocks_overriden: 5_000,
        };
        let meta = compute_leader_block_counts(&override_cfg, &override_cfg, &workers);
        assert_eq!(meta.num_host_blocks, 5_000);
        assert_eq!(meta.num_disk_blocks, 5_000);
    }

    #[test]
    fn tp_workers_sum_element_wise_before_sizing() {
        // Two TP workers each holding half of both groups. The element-wise
        // Σ reconstructs the full per-group sizes; the per-worker scalars
        // are deliberately wrong to prove the per-group path drives sizing.
        let half_groups = vec![6_422_528, 38_486_016];
        let workers = [
            worker(1_000, 1, half_groups.clone()),
            worker(1_000, 1, half_groups),
        ];
        let meta = compute_leader_block_counts(
            &window_cfg(WINDOW_BYTES),
            &window_cfg(WINDOW_BYTES),
            &workers,
        );
        assert_eq!(meta.num_host_blocks, 286);
        assert_eq!(meta.num_disk_blocks, 286);
    }

    #[test]
    fn legacy_peers_reproduce_the_scalar_formula() {
        // Pre-multigroup peers report empty per-group vecs; the leader falls
        // back to the scalar Σ — exactly today's formula.
        let workers = [worker(7_000_000, 4096, vec![])];
        let meta = compute_leader_block_counts(
            &window_cfg(24_500_000_000.0),
            &window_cfg(24_500_000_000.0),
            &workers,
        );
        assert_eq!(meta.num_host_blocks, 5_981_445); // floor(24.5e9 / 4096) = 5,981,445.3125
        assert_eq!(
            meta.num_host_blocks,
            compute_num_blocks(&window_cfg(24_500_000_000.0), 4096)
        );
        assert_eq!(meta.num_disk_blocks, 5_981_445);
    }

    #[test]
    fn mixed_legacy_and_modern_set_falls_back_to_scalar_sum() {
        // One legacy peer drags the element-wise aggregate to empty; the
        // scalar Σ (which the conservative writer always emits) drives
        // sizing instead — never a silent partial-group sum.
        let workers = [
            worker(7_000_000, 4_096, vec![]),
            worker(7_000_000, 8_192, vec![100, 200]),
        ];
        let meta = compute_leader_block_counts(
            &window_cfg(24_500_000_000.0),
            &window_cfg(24_500_000_000.0),
            &workers,
        );
        assert_eq!(meta.num_host_blocks, 1_993_815); // floor(24.5e9 / 12,288) = 1,993,815.104
        assert_eq!(meta.num_disk_blocks, 1_993_815);
    }
}

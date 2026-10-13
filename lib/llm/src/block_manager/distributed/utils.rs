// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

use crate::block_manager::connector::protocol::LeaderTransferRequest;

pub const ZMQ_PING_MESSAGE: &str = "ping";
pub const ZMQ_WORKER_METADATA_MESSAGE: &str = "worker_metadata";
pub const ZMQ_LEADER_METADATA_MESSAGE: &str = "leader_metadata";
pub const ZMQ_TRANSFER_BLOCKS_MESSAGE: &str = "transfer_blocks";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerMetadata {
    pub num_device_blocks: usize,
    pub bytes_per_block: usize,
    /// Multi-group: physical bytes-per-block per HMA layer group, in group
    /// registration order. Empty = legacy peer (a worker from before the
    /// hybrid-multigroup wire) — readers must fall back to `bytes_per_block`
    /// (the Σ scalar). Modern single-group workers send a 1-element vec; the
    /// empty-vs-single distinction is meaningful downstream, hence no
    /// `skip_serializing_if`.
    ///
    /// Compatibility coverage of `#[serde(default)]`: it rescues a missing
    /// field on self-describing decode paths (JSON) and covers the
    /// new-worker → old-leader direction on ZMQ (bincode ignores trailing
    /// bytes). KNOWN GAP: old-worker → new-leader over ZMQ is NOT covered —
    /// the handshake decodes with bincode positional
    /// (`decode_from_slice`, zmq.rs), so a 2-field legacy payload fails
    /// with UnexpectedEnd before the empty-vec fallback can apply.
    // Compatibility: workers < hybrid-multigroup send no field (N-2 window,
    // lib/llm/AGENTS.md). TODO(hybrid-multigroup): add a legacy-layout
    // fallback at the zmq.rs decode site, then remove this fallback when
    // N-2 no longer includes multigroup-less workers.
    #[serde(default)]
    pub bytes_per_block_per_group: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderMetadata {
    pub num_host_blocks: usize,
    pub num_disk_blocks: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Copy)]
pub enum BlockTransferPool {
    Device,
    Host,
    Disk,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ConnectorTransferType {
    Store,
    Load,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConnectorRequestLeader {
    pub req_id: String,
    pub txn_id: u64,
    pub transfer_type: ConnectorTransferType,
}

#[derive(Serialize, Deserialize, Debug, Getters, Clone, PartialEq, Eq)]
pub struct BlockTransferRequest {
    pub from_pool: BlockTransferPool,
    pub to_pool: BlockTransferPool,
    pub blocks: Vec<(usize, usize)>,

    /// Phase 2 multi-group: which HMA group's pools this request addresses.
    #[serde(default)]
    pub group: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector_req: Option<LeaderTransferRequest>,
}

impl BlockTransferRequest {
    #[allow(dead_code)]
    pub fn new(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            from_pool,
            to_pool,
            blocks,
            // Legacy constructor: single-group (group 0) semantics.
            group: 0,
            connector_req: None,
        }
    }

    pub fn new_with_trigger_id(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
        connector_req: LeaderTransferRequest,
    ) -> Self {
        Self {
            from_pool,
            to_pool,
            blocks,
            // Legacy constructor: single-group (group 0) semantics.
            group: 0,
            connector_req: Some(connector_req),
        }
    }
}

/// Element-wise Σ of per-group `bytes_per_block` across TP workers: group
/// `i` covers the same layers on every worker of a set, so the leader sums
/// each position across workers — the multi-group analog of the legacy
/// scalar Σ (`adjust policy for DP/PP if needed` still applies).
///
/// The result length is the shortest input. Modern workers of one set must
/// agree on the group count — a debug_assert fails dev/test builds loudly,
/// and release warns and degrades the set to the scalar path (empty result)
/// instead of silently truncating groups. A zero length is a legitimate
/// legacy peer, not a mismatch: any legacy peer truncates the result to
/// empty, which sends the caller to the legacy scalar path rather than
/// silently dropping that peer's contribution.
pub(crate) fn aggregate_worker_bytes_per_group<'a, I>(per_worker: I) -> Vec<usize>
where
    I: IntoIterator<Item = &'a [usize]>,
{
    let per_worker: Vec<&[usize]> = per_worker.into_iter().collect();
    let mut modern_group_counts: Vec<usize> = per_worker
        .iter()
        .filter(|groups| !groups.is_empty())
        .map(|groups| groups.len())
        .collect();
    modern_group_counts.sort_unstable();
    modern_group_counts.dedup();
    debug_assert!(
        modern_group_counts.len() <= 1,
        "workers disagree on the HMA group count (misconfigured TP set): got lengths {:?}",
        modern_group_counts
    );
    if modern_group_counts.len() > 1 {
        // Release-mode companion to the debug_assert: a misconfigured TP set
        // must not silently truncate to the shortest member's group count
        // (dropping groups). Degrade to the conservative scalar Σ (empty
        // result routes the caller to the legacy path) instead.
        tracing::warn!(
            "workers disagree on the HMA group count (misconfigured TP set): got lengths {:?}; \
             falling back to the scalar bytes-per-block sum",
            modern_group_counts
        );
        return Vec::new();
    }
    let Some(group_count) = per_worker.iter().map(|groups| groups.len()).min() else {
        return Vec::new();
    };
    (0..group_count)
        .map(|group| per_worker.iter().map(|groups| groups[group]).sum())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_json_without_per_group_field_deserializes_empty() {
        // THE N-2 contract (lib/llm/AGENTS.md): workers from before the
        // hybrid-multigroup wire send no `bytes_per_block_per_group`; the
        // tolerant reader must decode them to an empty vec (the legacy-peer
        // marker), never fail.
        let meta: WorkerMetadata =
            serde_json::from_str(r#"{"num_device_blocks":100,"bytes_per_block":4096}"#).unwrap();
        assert_eq!(meta.num_device_blocks, 100);
        assert_eq!(meta.bytes_per_block, 4096);
        assert_eq!(meta.bytes_per_block_per_group, Vec::<usize>::new());
    }

    #[test]
    fn multi_group_json_roundtrips_both_fields() {
        let meta: WorkerMetadata = serde_json::from_str(
            r#"{"num_device_blocks":100,"bytes_per_block":89817088,"bytes_per_block_per_group":[12845056,76972032]}"#,
        )
        .unwrap();
        assert_eq!(meta.bytes_per_block_per_group, vec![12_845_056, 76_972_032]);

        let encoded = serde_json::to_string(&meta).unwrap();
        // Conservative writer: the Σ scalar stays on the wire for legacy
        // readers, and the per-group vec roundtrips.
        assert!(encoded.contains(r#""bytes_per_block":89817088"#));
        assert!(encoded.contains(r#""bytes_per_block_per_group":[12845056,76972032]"#));
        let decoded: WorkerMetadata = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, meta);
    }

    #[test]
    fn single_group_worker_encodes_an_explicit_one_element_vec() {
        // No skip_serializing_if: empty vec (= legacy peer) and 1-element
        // vec (= modern single group) must stay distinguishable downstream.
        let meta = WorkerMetadata {
            num_device_blocks: 100,
            bytes_per_block: 4096,
            bytes_per_block_per_group: vec![4096],
        };
        let encoded = serde_json::to_string(&meta).unwrap();
        assert!(encoded.contains(r#""bytes_per_block_per_group":[4096]"#));
    }

    #[test]
    fn transfer_request_group_defaults_to_zero_and_roundtrips() {
        // THE N-2 contract (lib/llm/AGENTS.md): a legacy peer's request JSON
        // carries no `group` field; the tolerant reader must decode it to
        // group 0 (the single-group legacy pool set), never fail.
        let legacy: BlockTransferRequest = serde_json::from_str(
            r#"{"from_pool":"Device","to_pool":"Host","blocks":[[1,2],[3,4]]}"#,
        )
        .unwrap();
        assert_eq!(legacy.group, 0);
        assert_eq!(legacy.blocks, vec![(1, 2), (3, 4)]);

        // An explicit group roundtrips through the JSON transport the leader
        // uses for transfer requests.
        let grouped = BlockTransferRequest {
            from_pool: BlockTransferPool::Device,
            to_pool: BlockTransferPool::Host,
            blocks: vec![(5, 6)],
            group: 2,
            connector_req: None,
        };
        let encoded = serde_json::to_string(&grouped).unwrap();
        assert!(encoded.contains(r#""group":2"#));
        let decoded: BlockTransferRequest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, grouped);
    }

    #[test]
    fn bincode_roundtrip_matches_the_zmq_transport_encoding() {
        let meta = WorkerMetadata {
            num_device_blocks: 100,
            bytes_per_block: 89_817_088,
            bytes_per_block_per_group: vec![12_845_056, 76_972_032],
        };
        let bytes = bincode::serde::encode_to_vec(&meta, bincode::config::standard()).unwrap();
        let (decoded, _) = bincode::serde::decode_from_slice::<WorkerMetadata, _>(
            &bytes,
            bincode::config::standard(),
        )
        .unwrap();
        assert_eq!(decoded, meta);
    }

    #[test]
    fn aggregation_sums_element_wise_across_tp_workers() {
        let per_worker = [vec![100, 200], vec![100, 200]];
        assert_eq!(
            aggregate_worker_bytes_per_group(per_worker.iter().map(|v| v.as_slice())),
            vec![200, 400]
        );
    }

    #[test]
    fn aggregation_of_a_single_worker_is_itself() {
        let per_worker = [vec![100, 200]];
        assert_eq!(
            aggregate_worker_bytes_per_group(per_worker.iter().map(|v| v.as_slice())),
            vec![100, 200]
        );
    }

    #[test]
    fn aggregation_of_all_legacy_peers_is_empty() {
        let per_worker = [Vec::<usize>::new(), Vec::<usize>::new()];
        assert_eq!(
            aggregate_worker_bytes_per_group(per_worker.iter().map(|v| v.as_slice())),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn one_legacy_peer_downgrades_the_whole_set() {
        // Mixed-version TP set during an N-2 rolling update: the legacy
        // peer's empty vec truncates the element-wise sum to zero groups, so
        // the caller falls back to the scalar Σ every version emits — never
        // a silent partial-group sum.
        let per_worker = [Vec::<usize>::new(), vec![100, 200]];
        assert_eq!(
            aggregate_worker_bytes_per_group(per_worker.iter().map(|v| v.as_slice())),
            Vec::<usize>::new()
        );
    }
}

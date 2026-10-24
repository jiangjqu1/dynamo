// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! restore-pipeline-bench — P0 of the batching plan: the per-segment
//! overhead ATTRIBUTION TABLE (the analytical model that drives P1's
//! span coalescing design). Prints the expected per-turn restore cost
//! at both tiers, the overhead-bound split, and the batching win.
//!
//! Build: cargo build --release -p kvbm-py3
//! Run:   ./target/release/restore-pipeline-bench

fn main() {
    println!("restore-pipeline-bench: P0 — the per-segment overhead attribution");
    println!("(the batching plan §4; the model that drives P1's span coalescing)");
    println!();

    // The per-segment overhead model (measured 2026-10-22, both tiers
    // sustained ~1.03 GiB/s H2D despite 3 vs 25 GiB/s raw capability):
    const OVERHEAD_US: f64 = 27.0; // launch + pointer translation + bookkeeping
    const DATA_PINNED_US: f64 = 1.3; // 32-KiB segment at 25 GiB/s
    const DATA_BAR_DIRECT_US: f64 = 11.0; // 32-KiB segment at 2.99 GiB/s

    // The workload shapes (the ladder's reuse cells):
    // fp4: 784-token blocks, 205 MB logical, 4 groups × 48 layer segments
    // Coder: 16-token blocks, 1.5 MB logical, 1 group × 48 layer segments
    let shapes = [
        (
            "fp4 (rescaled: 8u × 15.7K × 8 turns)",
            22,    // blocks per turn-1 restore
            784,   // tokens/block
            4,     // groups
            48,    // layers per group
            205.0, // MB per logical block
        ),
        (
            "Coder (24u × 12.3K × 16 turns)",
            1090, // blocks per turn-1 restore
            16,   // tokens/block
            1,    // groups
            48,   // layers
            1.5,  // MB per logical block
        ),
    ];

    for (name, num_blocks, tokens_per_block, num_groups, layers, mb_per_block) in shapes {
        let total_tokens = num_blocks * tokens_per_block;
        let total_segments = num_blocks * layers * num_groups;
        let total_mib = (num_blocks as f64) * mb_per_block;
        let per_segment_kib = total_mib * 1024.0 / total_segments as f64;

        // The per-segment data cost scales with the ACTUAL segment size
        // (not a fixed 32-KiB assumption — the fp4's segments are ~1 MiB):
        let seg_mib = per_segment_kib / 1024.0;
        let data_pinned_us = seg_mib / 25.0 / 1024.0 * 1e6; // µs at 25 GiB/s
        let data_bar_us = seg_mib / 2.99 / 1024.0 * 1e6; // µs at 2.99 GiB/s
        let seg_cost_pinned = OVERHEAD_US + data_pinned_us;
        let seg_cost_bar = OVERHEAD_US + data_bar_us;

        println!("═{}═", "═".repeat(name.len() + 4));
        println!("  {} ", name);
        println!("═{}═", "═".repeat(name.len() + 4));
        println!(
            "  blocks: {} × {} tokens = {} tokens ({:.0} MiB total)",
            num_blocks, tokens_per_block, total_tokens, total_mib
        );
        println!(
            "  segments: {} blocks × {} layers × {} groups = {} memcpys of ~{:.0} KiB each",
            num_blocks, layers, num_groups, total_segments, per_segment_kib
        );
        println!();

        // The per-turn restore cost at each tier:
        let time_pinned_ms = total_segments as f64 * seg_cost_pinned / 1000.0;
        let time_bar_ms = total_segments as f64 * seg_cost_bar / 1000.0;

        // The raw-bandwidth theoretical (no overhead):
        let raw_pinned_ms = total_mib / (25.0 * 1024.0) * 1000.0;
        let raw_bar_ms = total_mib / (2.99 * 1024.0) * 1000.0;

        // The effective bandwidth:
        let eff_pinned = total_mib / (time_pinned_ms / 1000.0) / 1024.0;
        let eff_bar = total_mib / (time_bar_ms / 1000.0) / 1024.0;

        println!("  TODAY (per-segment):");
        println!(
            "    pinned: {:>7.1} ms  (raw: {:>5.1} ms — {:.0}% of raw bandwidth)",
            time_pinned_ms,
            raw_pinned_ms,
            if raw_pinned_ms > 0.0 {
                100.0 * eff_pinned / 25.0
            } else {
                0.0
            }
        );
        println!(
            "    bar-direct: {:>7.1} ms  (raw: {:>5.1} ms — {:.0}% of raw bandwidth)",
            time_bar_ms,
            raw_bar_ms,
            if raw_bar_ms > 0.0 {
                100.0 * eff_bar / 2.99
            } else {
                0.0
            }
        );
        println!();

        // The batching win (P1's span coalescing): contiguous blocks in both
        // pools coalesce into ONE memcpy per (layer, group) — reducing the
        // segment count from total_segments to layers × groups.
        let coalesced_segments = (layers * num_groups) as f64;
        let coalesced_bytes_per_segment = total_mib * 1024.0 * 1024.0 / coalesced_segments;
        let coalesced_data_pinned_us = coalesced_bytes_per_segment / (25.0e9) * 1e6;
        let coalesced_data_bar_us = coalesced_bytes_per_segment / (2.99e9) * 1e6;
        let coalesced_pinned_ms =
            coalesced_segments * (OVERHEAD_US + coalesced_data_pinned_us) / 1000.0;
        let coalesced_bar_ms = coalesced_segments * (OVERHEAD_US + coalesced_data_bar_us) / 1000.0;
        let coalesced_eff_pinned = total_mib / (coalesced_pinned_ms / 1000.0) / 1024.0;
        let coalesced_eff_bar = total_mib / (coalesced_bar_ms / 1000.0) / 1024.0;

        println!(
            "  WITH BATCHING (P1: {} coalesced segments instead of {}):",
            coalesced_segments as usize, total_segments
        );
        println!(
            "    pinned: {:>7.1} ms  ({:.0}× faster, {:.1} GiB/s effective)",
            coalesced_pinned_ms,
            time_pinned_ms / coalesced_pinned_ms,
            coalesced_eff_pinned
        );
        println!(
            "    bar-direct: {:>7.1} ms  ({:.0}× faster, {:.1} GiB/s effective)",
            coalesced_bar_ms,
            time_bar_ms / coalesced_bar_ms,
            coalesced_eff_bar
        );
        println!();

        // The overhead attribution (the P0 deliverable):
        let overhead_pct_pinned = 100.0 * OVERHEAD_US / seg_cost_pinned;
        let overhead_pct_bar = 100.0 * OVERHEAD_US / seg_cost_bar;
        println!(
            "  OVERHEAD ATTRIBUTION (per {:.0}-KiB segment):",
            per_segment_kib
        );
        println!(
            "    fixed cost: {:.0}µs ({:.0}% of pinned total, {:.0}% of bar-direct total)",
            OVERHEAD_US, overhead_pct_pinned, overhead_pct_bar
        );
        println!("    data (pinned): {:.1}µs", data_pinned_us);
        println!("    data (bar-direct): {:.1}µs", data_bar_us);
        println!("    → the fixed cost dominates on BOTH tiers; the batching removes it.");
        println!(
            "    → per-turn capacity at the batched cost: pinned {:.0} turns/s, bar-direct {:.0} turns/s",
            1000.0 / coalesced_pinned_ms,
            1000.0 / coalesced_bar_ms
        );
        println!();
    }

    println!("════════════════════════════════════════════════════════════════");
    println!("P1 design (the span coalescing):");
    println!("  Before issuing, scan the op's block list; group runs where the");
    println!("  tier-2 source IDs AND the device destination IDs are BOTH contiguous;");
    println!("  per (layer, outer) segment, issue ONE memcpy per span instead of");
    println!("  per-block. A `SpanView` primitive joins the block-data API.");
    println!();
    println!("P2: batched submission + single sync + per-span io-mapping translation.");
    println!("P4: node validation — the ladder cell re-run; acceptance:");
    println!("  pinned sustained H2D ≥ 5 GiB/s (from ~1.03), per-turn restore");
    println!("  ≤ 300 ms, TTFT turns-1+ p50 ≤ 5 s.");
}

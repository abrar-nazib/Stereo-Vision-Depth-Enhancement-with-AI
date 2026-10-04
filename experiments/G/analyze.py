"""Paired, source-frame-grouped post-run comparison against frozen E3."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from experiments.B.B_0_fused_baseline.run import DATA, split_rows
from experiments.G.model import ARMS, EdgeHead
from experiments.G.run import FrozenE3, ROOT, evaluate


RUN_ID = "g_edge_native_seed42_10k_v2_20261004"
OUTPUT = ROOT / "experiments/G/runs"


def main() -> None:
    rows = split_rows(json.loads((DATA / "manifest.json").read_text()), 42)[2]
    torch.manual_seed(42)
    baseline = evaluate(FrozenE3(), EdgeHead("G_1_rgb_residual").cuda(), rows, DATA)
    pair_lookup = {(p["scene"], p["variation"], p["frame"]): p
                   for p in baseline["pair_results"]}
    analysis = {"run_id": RUN_ID, "baseline": {k: v for k, v in baseline.items()
                                                 if k != "pair_results"}, "arms": {}}
    for arm in ARMS:
        result = json.loads((OUTPUT / arm / RUN_ID / "test.json").read_text())
        by_frame: dict[int, list[tuple[float, int]]] = defaultdict(list)
        improvements = 0
        for pair in result["pair_results"]:
            key = (pair["scene"], pair["variation"], pair["frame"])
            base = pair_lookup[key]
            delta = pair["epe"] - base["epe"]
            by_frame[pair["frame"]].append((delta, pair["n"]))
            improvements += delta < 0
        frame_delta = {str(frame): sum(delta * n for delta, n in values) / sum(n for _, n in values)
                       for frame, values in sorted(by_frame.items())}
        analysis["arms"][arm] = {
            "test_epe_delta_px": result["epe"] - baseline["epe"],
            "test_edge_epe_delta_px": result["edge_epe"] - baseline["edge_epe"],
            "improved_pairs_of_100": improvements,
            "improved_source_frames_of_10": sum(delta < 0 for delta in frame_delta.values()),
            "source_frame_epe_delta_px": frame_delta,
        }
    destination = ROOT / "experiments/G/paired_analysis.json"
    destination.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({"baseline_epe": baseline["epe"],
                      "baseline_edge_epe": baseline["edge_epe"],
                      "arms": analysis["arms"]}, indent=2))


if __name__ == "__main__":
    main()

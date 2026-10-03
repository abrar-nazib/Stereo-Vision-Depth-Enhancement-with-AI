"""Frozen F1/F2/F3 evaluation on KITTI 2015's public labeled pairs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from experiments.B.B_0_fused_baseline.model import FusionStereoLite
from experiments.B.B_0_fused_baseline.run import pad32, sha256
from experiments.C.C_1_large_control.run import atomic_json
from experiments.E.E_1_wide2_semantics.model import EModel
from experiments.F.kitti_data import list_pair_ids, read_pair
from experiments.vkitti2.run import DisparityMeter


def pair_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    labeled = torch.isfinite(target) & (target > 0)
    evaluable = labeled & (target < 192)
    meter = DisparityMeter()
    meter.add(prediction, target)
    return {**meter.result(), "gt_labeled_pixels": int(labeled.sum()),
            "evaluated_pixels": int(evaluable.sum()),
            "out_of_range_pixels": int((labeled & ~evaluable).sum())}


def _model(arm: str, stereo: Path, semantic: Path, encoder: Path, head: Path | None):
    if arm == "F1":
        model = FusionStereoLite("V", encoder=encoder)
        payload = torch.load(stereo, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
    elif arm in ("F2", "F3"):
        if head is None:
            raise ValueError(f"{arm} requires its selected head checkpoint")
        expected_head = "E3" if arm == "F2" else "E4"
        model = EModel(stereo, semantic, encoder, gate_hidden=32,
                       residual_hidden=128, use_semantics=(arm == "F2"))
        payload = torch.load(head, map_location="cpu", weights_only=False)
        manifest = payload["manifest"]
        if (manifest["arm"] != expected_head or
                manifest["use_semantics"] != (arm == "F2") or
                manifest["stereo_sha256"] != sha256(stereo) or
                manifest["semantic_sha256"] != sha256(semantic) or
                manifest["encoder_sha256"] != sha256(encoder)):
            raise ValueError(f"{expected_head} head provenance differs from the frozen predictors")
        model.load_trainable_state(payload["trainable"])
    else:
        raise ValueError(f"unknown F arm: {arm}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.cuda().eval()


@torch.inference_mode()
def evaluate(arm: str, archive_path: Path, stereo: Path, semantic: Path,
             encoder: Path, head: Path | None, output: Path,
             limit: int | None = None) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("F-series evaluation requires CUDA")
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        ids = list_pair_ids(archive)
    if len(ids) != 200:
        raise ValueError(f"expected 200 KITTI 2015 labeled pairs; found {len(ids)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        ids = ids[:limit]
    model = _model(arm, stereo, semantic, encoder, head)
    meters = {kind: DisparityMeter() for kind in ("occ", "noc")}
    counts = {kind: {"gt_labeled_pixels": 0, "evaluated_pixels": 0,
                     "out_of_range_pixels": 0} for kind in meters}
    pair_results = []
    latencies_ms = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with ZipFile(archive_path) as archive:
        for index, stem in enumerate(ids, start=1):
            left, right, occ, noc = read_pair(archive, stem)
            def tensor(image: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].cuda().float()
            lp, top = pad32(tensor(left))
            rp, _ = pad32(tensor(right))
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(lp, rp)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            torch.cuda.synchronize()
            latency = (time.perf_counter() - tick) * 1000
            latencies_ms.append(latency)
            prediction = prediction[..., top:top + occ.shape[0], :occ.shape[1]]
            metrics = {}
            for kind, gt in (("occ", occ), ("noc", noc)):
                target = torch.from_numpy(gt)[None, None].cuda()
                result = pair_metrics(prediction, target)
                metrics[kind] = result
                meters[kind].add(prediction, target)
                for key in counts[kind]:
                    counts[kind][key] += result[key]
            pair_results.append({"id": stem, "occ": metrics["occ"],
                                 "noc": metrics["noc"], "inference_ms": latency})
            if index % 20 == 0 or index == len(ids):
                atomic_json(output / "status.json", {"state": "running", "arm": arm,
                                                    "pairs": index, "total": len(ids)})
                print(json.dumps({"arm": arm, "pairs": index, "total": len(ids)}), flush=True)
    measured = latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms
    result = {"arm": arm, "dataset": "KITTI 2015 labeled training pairs",
              "pairs": len(ids), "smoke_limit": limit,
              "preprocessing": "RGB 0..255; native resolution; replicate-pad top/right to /32; no resize",
              "valid_mask": "GT finite, >0 and <192 px; KITTI PNG units /256",
              "checkpoint_step": (int(torch.load(head, map_location="cpu", weights_only=False)["step"])
                                  if arm in ("F2", "F3") else None),
              "occ": {**meters["occ"].result(), **counts["occ"]},
              "noc": {**meters["noc"].result(), **counts["noc"]},
              "latency_ms_median": statistics.median(measured),
              "latency_ms_mean": statistics.mean(measured),
              "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
              "elapsed_s": time.perf_counter() - started,
              "pair_results": pair_results}
    atomic_json(output / "result.json", result)
    atomic_json(output / "status.json", {"state": "complete", "arm": arm,
                                        "pairs": len(ids), "occ_epe": result["occ"]["epe"]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("F1", "F2", "F3"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--stereo", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    evaluate(args.arm, args.archive, args.stereo, args.semantic, args.encoder,
             args.head, args.output, args.limit)


if __name__ == "__main__":
    main()

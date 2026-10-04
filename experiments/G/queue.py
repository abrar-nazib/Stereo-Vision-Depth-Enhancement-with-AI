"""Run G1/G2/G3 sequentially on the local RTX 3050 with durable per-arm logs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.B.B_0_fused_baseline.run import DATA
from experiments.G.model import ARMS


ROOT = Path(__file__).resolve().parents[2]


def write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="g_edge_native_seed42_10k_v2_20261004")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    args = parser.parse_args()
    if not (DATA / "manifest.json").is_file():
        raise FileNotFoundError(DATA / "manifest.json")
    directory = ROOT / "experiments/G/queue" / args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    status = directory / "status.json"
    completed: list[str] = []
    write_status(status, {"state": "queued", "arms": ARMS, "completed": completed,
                          "run_id": args.run_id, "started_unix": time.time()})
    for arm in ARMS:
        command = [sys.executable, "-m", "experiments.G.run", "--arm", arm,
                   "--run-id", args.run_id, "--steps", str(args.steps),
                   "--eval-every", str(args.eval_every)]
        log = directory / f"{arm}.log"
        write_status(status, {"state": "running", "current_arm": arm,
                              "completed": completed, "log": str(log),
                              "command": command, "updated_unix": time.time()})
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            write_status(status, {"state": "failed", "current_arm": arm,
                                  "completed": completed, "exit_code": result.returncode,
                                  "log": str(log), "updated_unix": time.time()})
            raise SystemExit(result.returncode)
        completed.append(arm)
        write_status(status, {"state": "running", "completed": completed,
                              "updated_unix": time.time()})
    write_status(status, {"state": "complete", "completed": completed,
                          "run_id": args.run_id, "updated_unix": time.time()})


if __name__ == "__main__":
    main()

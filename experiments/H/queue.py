"""Run matched 25k-step stereo-only ablations sequentially on the local GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.H.model import ARMS
from experiments.H.run import DATA


ROOT = Path(__file__).resolve().parents[2]


def status_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="h_stereo_only_2k_25k_scratchhead_seed42_v2")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--eval-every", type=int, default=1000)
    args = parser.parse_args()
    if not (DATA / "manifest.json").is_file():
        raise FileNotFoundError(DATA / "manifest.json")
    directory = ROOT / "experiments/H/queue" / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    status = directory / "status.json"
    completed = []
    for arm in ARMS:
        run_dir = ROOT / "experiments/H/runs" / arm / args.run_id
        if (run_dir / "summary.json").is_file():
            completed.append(arm)
            continue
        command = [sys.executable, "-m", "experiments.H.run", "--arm", arm,
                   "--run-id", args.run_id, "--steps", str(args.steps),
                   "--eval-every", str(args.eval_every),
                   "--initialization", "scratch"]
        log = directory / f"{arm}.log"
        status_write(status, {"state": "running", "current_arm": arm,
                              "completed": completed, "log": str(log),
                              "command": command, "updated_unix": time.time()})
        with log.open("a") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            status_write(status, {"state": "failed", "current_arm": arm,
                                  "completed": completed, "exit_code": result.returncode,
                                  "log": str(log), "updated_unix": time.time()})
            raise SystemExit(result.returncode)
        completed.append(arm)
    status_write(status, {"state": "complete", "completed": completed,
                          "run_id": args.run_id, "updated_unix": time.time()})


if __name__ == "__main__":
    main()

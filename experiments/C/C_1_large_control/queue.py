"""Sequential detached-friendly launcher for C-series heads and controls."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.C.C_1_large_control.run import ARMS, CONTROL_ARMS, ROOT, atomic_json


def queue_selection(controls: bool) -> tuple[list[str], str]:
    if controls:
        return list(CONTROL_ARMS), "C_4_no_semantics"
    return list(ARMS), "C_1_large_control"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="c_v1_seed42_20261001")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--controls", action="store_true",
                        help="Queue the two equal-architecture C2 causal controls")
    args = parser.parse_args()
    arms, folder = queue_selection(args.controls)
    queue_dir = ROOT / "experiments" / "C" / folder / "queue" / args.run_id
    queue_dir.mkdir(parents=True, exist_ok=False)
    status = queue_dir / "status.json"
    atomic_json(status, {"state": "queued", "run_id": args.run_id,
                         "arms": arms, "completed": [], "started_unix": time.time()})
    completed = []
    for arm in arms:
        command = [sys.executable, "-m", "experiments.C.C_1_large_control.run",
                   "--arm", arm, "--run-id", args.run_id,
                   "--steps", str(args.steps), "--eval-every", str(args.eval_every),
                   "--patience", str(args.patience)]
        atomic_json(status, {"state": "running", "run_id": args.run_id,
                             "current_arm": arm, "completed": completed,
                             "command": command, "updated_unix": time.time()})
        log_path = queue_dir / f"{arm}.log"
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            atomic_json(status, {"state": "failed", "run_id": args.run_id,
                                 "current_arm": arm, "completed": completed,
                                 "exit_code": result.returncode,
                                 "log": str(log_path), "updated_unix": time.time()})
            print(json.dumps({"state": "failed", "arm": arm, "log": str(log_path)}), flush=True)
            raise SystemExit(result.returncode)
        completed.append(arm)
        print(json.dumps({"state": "arm_complete", "arm": arm,
                          "log": str(log_path)}), flush=True)
    atomic_json(status, {"state": "complete", "run_id": args.run_id,
                         "completed": completed, "updated_unix": time.time()})


if __name__ == "__main__":
    main()

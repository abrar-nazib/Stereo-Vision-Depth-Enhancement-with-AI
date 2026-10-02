"""Launch all D arms sequentially on one local GPU; safe for nohup detachment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from experiments.C.C_1_large_control.run import atomic_json
from experiments.D.D_1_semantic_cost.run import ARMS, DATA, ROOT, STEREO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="d_v1_seed42_20261002")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if not DATA.joinpath("manifest.json").is_file() or not STEREO.is_file():
        raise FileNotFoundError("SSD dataset/checkpoint missing; queue not started")
    arms = list(ARMS)
    queue_dir = ROOT / "experiments/D/D_1_semantic_cost/queue" / args.run_id
    queue_dir.mkdir(parents=True, exist_ok=False)
    status = queue_dir / "status.json"
    atomic_json(status, {"state": "queued", "arms": arms, "completed": [],
                         "run_id": args.run_id, "started_unix": time.time()})
    completed = []
    for arm in arms:
        command = [sys.executable, "-m", "experiments.D.D_1_semantic_cost.run",
                   "--arm", arm, "--run-id", args.run_id,
                   "--steps", str(args.steps), "--eval-every", str(args.eval_every),
                   "--patience", str(args.patience)]
        log_path = queue_dir / f"{arm}.log"
        atomic_json(status, {"state": "running", "run_id": args.run_id,
                             "current_arm": arm, "completed": completed,
                             "log": str(log_path), "command": command,
                             "updated_unix": time.time()})
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            atomic_json(status, {"state": "failed", "run_id": args.run_id,
                                 "current_arm": arm, "completed": completed,
                                 "exit_code": result.returncode, "log": str(log_path),
                                 "updated_unix": time.time()})
            print(json.dumps({"state": "failed", "arm": arm, "log": str(log_path)}), flush=True)
            raise SystemExit(result.returncode)
        completed.append(arm)
        print(json.dumps({"state": "arm_complete", "arm": arm,
                          "log": str(log_path)}), flush=True)
    atomic_json(status, {"state": "complete", "run_id": args.run_id,
                         "completed": completed, "updated_unix": time.time()})


if __name__ == "__main__":
    main()

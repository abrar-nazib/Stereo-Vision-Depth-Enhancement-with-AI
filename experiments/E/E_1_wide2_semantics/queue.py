"""Run the four E arms sequentially on one local GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from experiments.B.B_0_fused_baseline.run import DATA, ROOT, SEMANTIC, STEREO
from experiments.C.C_1_large_control.run import atomic_json
from experiments.E.E_1_wide2_semantics.run import ARM_CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="e_capacity_v1_seed42_20261002")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.patience) < 1:
        parser.error("steps, eval-every and patience must be positive")

    queue_dir = ROOT / "experiments/E/E_1_wide2_semantics/queue" / args.run_id
    jobs = [
        {
            "arm": arm,
            "command": [sys.executable, "-m", "experiments.E.E_1_wide2_semantics.run",
                        "--arm", arm, "--run-id", args.run_id,
                        "--steps", str(args.steps), "--eval-every", str(args.eval_every),
                        "--patience", str(args.patience)],
            "log": str(queue_dir / f"{arm}.log"),
        }
        for arm in ARM_CONFIGS
    ]
    if args.dry_run:
        print(json.dumps({"run_id": args.run_id, "jobs": jobs}))
        return
    if not DATA.joinpath("manifest.json").is_file() or not STEREO.is_file() or not SEMANTIC.is_file():
        raise FileNotFoundError("VKITTI subset, stereo checkpoint or semantic checkpoint missing")

    queue_dir.mkdir(parents=True, exist_ok=False)
    status = queue_dir / "status.json"
    completed: list[str] = []
    atomic_json(status, {"state": "queued", "run_id": args.run_id,
                         "arms": list(ARM_CONFIGS), "completed": completed,
                         "started_unix": time.time()})
    for job in jobs:
        atomic_json(status, {"state": "running", "run_id": args.run_id,
                             "current_arm": job["arm"], "completed": completed,
                             "log": job["log"], "command": job["command"],
                             "updated_unix": time.time()})
        with open(job["log"], "w") as log:
            result = subprocess.run(job["command"], cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            atomic_json(status, {"state": "failed", "run_id": args.run_id,
                                 "current_arm": job["arm"], "completed": completed,
                                 "exit_code": result.returncode, "log": job["log"],
                                 "updated_unix": time.time()})
            raise SystemExit(result.returncode)
        completed.append(job["arm"])
        print(json.dumps({"state": "arm_complete", "arm": job["arm"],
                          "log": job["log"]}), flush=True)
    atomic_json(status, {"state": "complete", "run_id": args.run_id,
                         "completed": completed, "updated_unix": time.time()})


if __name__ == "__main__":
    main()

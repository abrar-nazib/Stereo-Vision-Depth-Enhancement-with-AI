"""E-series width sweep; D runner preserves the 800/100/100 protocol."""

from __future__ import annotations

import json
import sys

from experiments.B.B_0_fused_baseline.run import ADE, SEMANTIC, STEREO
from experiments.D.D_1_semantic_cost.run import main as run_d_protocol
from experiments.E.E_1_wide2_semantics.model import EModel


ARM_CONFIGS = {
    "E_1_wide2_semantics": {"gate_hidden": 16, "residual_hidden": 64,
                            "use_semantics": True},
    "E_2_wide2_control": {"gate_hidden": 16, "residual_hidden": 64,
                          "use_semantics": False},
    "E_3_wide4_semantics": {"gate_hidden": 32, "residual_hidden": 128,
                            "use_semantics": True},
    "E_4_wide4_control": {"gate_hidden": 32, "residual_hidden": 128,
                          "use_semantics": False},
}


def construct(arm: str) -> EModel:
    return EModel(STEREO, SEMANTIC, ADE, **ARM_CONFIGS[arm])


def main() -> None:
    if sys.argv[1:] == ["--list-arms"]:
        print(json.dumps(ARM_CONFIGS, sort_keys=True))
        return
    run_d_protocol(arms=ARM_CONFIGS, factory=construct, series="E")


if __name__ == "__main__":
    main()

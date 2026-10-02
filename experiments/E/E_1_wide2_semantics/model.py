"""Capacity-only variants of D2's frozen-predictor semantic gate and refiner."""

from experiments.D.D_1_semantic_cost.model import (
    ClassResidual,
    DModel,
    SemanticCostGate,
)


class EModel(DModel):
    def __init__(self, stereo_checkpoint, semantic_checkpoint, encoder_checkpoint,
                 *, gate_hidden: int, residual_hidden: int, use_semantics: bool):
        super().__init__(stereo_checkpoint, semantic_checkpoint, encoder_checkpoint,
                         refinement=True, use_semantics=use_semantics)
        self.gate = SemanticCostGate(hidden_channels=gate_hidden)
        self.refiner = ClassResidual(hidden_channels=residual_hidden)
        self.architecture_name = "D2 candidate gate and class residual, width-scaled only"
        self.ablation_metadata = {
            "gate_hidden": gate_hidden,
            "residual_hidden": residual_hidden,
            "use_semantics": use_semantics,
            "reference": "D2/D4 1x, same frozen predictors and training protocol",
        }

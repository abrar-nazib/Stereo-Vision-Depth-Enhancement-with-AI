"""Contract tests for the B-series single-trunk stereo/semantic model."""

from pathlib import Path

import pytest
import torch
from ultralytics import YOLO

from experiments.B.B_0_fused_baseline.model import FusedStereoSemantic
from experiments.B.B_0_fused_baseline.predict import FusedPredictor, dense_uvdcc


ROOT = Path(__file__).resolve().parents[1]
STEREO = Path("/media/abrar/AbrarSSD/ResearchArtifacts/SVDE/final_pass/"
              "a09m_fullsf_a10_v1_20260929/checkpoints/best.pth")
SEMANTIC = ROOT / "models/segmentation/yolo26m-sem-vkitti2-14class-freeze7-best.pt"
ADE = ROOT / "models/segmentation/yolo26m-sem-ade20k.pt"


@pytest.mark.skipif(not all(p.exists() for p in (STEREO, SEMANTIC, ADE)),
                    reason="local research checkpoints unavailable")
def test_fused_matches_legacy_and_runs_shared_trunk_once():
    from experiments.vkitti2.run import FusionStereoLite

    torch.manual_seed(7)
    model = FusedStereoSemantic(STEREO, SEMANTIC, ADE).eval()
    left = torch.rand(1, 3, 64, 96) * 255
    right = torch.rand_like(left) * 255
    calls = []
    hook = model.shared_layers[0].register_forward_hook(
        lambda _module, _inputs, _outputs: calls.append(1))
    with torch.inference_mode():
        actual_d, actual_s = model(left, right)
    hook.remove()
    assert len(calls) == 1

    stereo = FusionStereoLite("V", encoder=ADE).eval()
    stereo.load_state_dict(torch.load(STEREO, map_location="cpu", weights_only=False)["model"])
    semantic = YOLO(str(SEMANTIC)).model.eval()
    with torch.inference_mode():
        expected_d = stereo(left, right)
        expected_s = semantic(left / 255.0)
    torch.testing.assert_close(actual_d, expected_d, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_s, expected_s, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not all(p.exists() for p in (STEREO, SEMANTIC, ADE)),
                    reason="local research checkpoints unavailable")
def test_fused_exposes_detached_multiscale_features_without_changing_outputs():
    model = FusedStereoSemantic(STEREO, SEMANTIC, ADE).eval()
    left = torch.rand(1, 3, 64, 96) * 255
    right = torch.rand_like(left)
    with torch.inference_mode():
        expected = model(left, right)
        disparity, logits, features = model(left, right, return_features=True)
    torch.testing.assert_close(disparity, expected[0])
    torch.testing.assert_close(logits, expected[1])
    assert features["left_f4"].shape[1:] == (256, 16, 24)
    assert features["left_f8"].shape[1:] == (512, 8, 12)
    assert features["right_f8"].shape == features["left_f8"].shape
    assert features["tile_f4"].shape[1:] == (16, 16, 24)
    assert features["tile_conf4"].shape[1:] == (1, 16, 24)
    assert features["semantic_f8"].shape[1:] == (256, 8, 12)


@pytest.mark.skipif(not all(p.exists() for p in (STEREO, SEMANTIC, ADE)),
                    reason="local research checkpoints unavailable")
def test_mismatched_shared_trunk_is_rejected():
    with pytest.raises(ValueError, match="shared trunk"):
        FusedStereoSemantic(STEREO, SEMANTIC, ROOT / "models/segmentation/yolo26s-sem-ade20k.pt")


@pytest.mark.skipif(not torch.cuda.is_available() or not all(
    p.exists() for p in (STEREO, SEMANTIC, ADE)), reason="CUDA or local checkpoints unavailable")
def test_cuda_amp_fused_matches_legacy():
    from experiments.vkitti2.run import FusionStereoLite

    torch.manual_seed(11)
    left = (torch.rand(1, 3, 128, 160, device="cuda") * 255)
    right = (torch.rand_like(left) * 255)
    fused = FusedStereoSemantic(STEREO, SEMANTIC, ADE).cuda().eval()
    stereo = FusionStereoLite("V", encoder=ADE).cuda().eval()
    stereo.load_state_dict(torch.load(STEREO, map_location="cpu", weights_only=False)["model"])
    semantic = YOLO(str(SEMANTIC)).model.cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        actual_d, actual_s = fused(left, right)
        expected_d, expected_s = stereo(left, right), semantic(left / 255.0)
    torch.testing.assert_close(actual_d.float(), expected_d.float(), rtol=0.002, atol=0.03)
    torch.testing.assert_close(actual_s.float(), expected_s.float(), rtol=0.002, atol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available() or not all(
    p.exists() for p in (STEREO, SEMANTIC, ADE)), reason="CUDA or local checkpoints unavailable")
def test_cuda_parallel_branches_match_single_stream():
    model = FusedStereoSemantic(STEREO, SEMANTIC, ADE).cuda().eval()
    left = torch.rand(1, 3, 128, 160, device="cuda") * 255
    right = torch.rand_like(left) * 255
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        sequential = model(left, right)
        parallel = model(left, right, parallel=True)
    torch.cuda.synchronize()
    for expected, actual in zip(sequential, parallel):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not all(p.exists() for p in (STEREO, SEMANTIC, ADE)),
                    reason="local research checkpoints unavailable")
def test_prediction_contract_unpads_and_provides_class_confidence():
    model = FusedStereoSemantic(STEREO, SEMANTIC, ADE).eval()
    predictor = FusedPredictor(model)
    left = torch.rand(1, 3, 61, 91) * 255
    right = torch.rand_like(left) * 255
    with torch.inference_mode():
        result = predictor(left, right)
    assert result["disparity_px"].shape == (1, 1, 61, 91)
    assert result["class_id"].shape == (1, 1, 61, 91)
    assert result["class_confidence"].shape == (1, 1, 61, 91)
    assert result["class_id"].dtype == torch.long
    assert int(result["class_id"].min()) >= 0
    assert int(result["class_id"].max()) < 14
    assert float(result["class_confidence"].min()) >= 0
    assert float(result["class_confidence"].max()) <= 1
    records = dense_uvdcc(result)
    assert records.shape == (1, 61, 91, 5)
    torch.testing.assert_close(records[0, 4, 7, :2], torch.tensor([7.0, 4.0]))

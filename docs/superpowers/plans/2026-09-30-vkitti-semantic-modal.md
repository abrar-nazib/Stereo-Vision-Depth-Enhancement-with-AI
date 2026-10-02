# VKITTI2 Frozen-Encoder Semantic Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the existing YOLO26m-sem ADE20K checkpoint on full VKITTI2 class masks using Modal A10 while preserving the exact layers 0–6 consumed by the trained stereo head.

**Architecture:** A CPU preparation stage audits and converts VKITTI RGB class masks to indexed PNGs, with all variations of a scene kept in one split. An A10 Ultralytics semantic trainer uses the pretrained model, freezes layers 0–6 including BatchNorm statistics, verifies their state against the source weights, and stores checkpoints/metrics on a Modal volume. A short A10 probe precedes a detached full run.

**Tech Stack:** Python 3.12, project-wide uv, Modal, Ultralytics semantic task, PyTorch, OpenCV/Numpy.

**Spec:** `experiments/vkitti2/RESEARCH_DIRECTION.md`; user-approved scene split and frozen-trunk design (2026-09-30).

## Global Constraints

- Use `svde-vkitti2` source archives and `svde-results` for outputs. Extract large archives on container-local disk, not the Modal Volume.
- Left RGB (`Camera_0`) and class masks only; no depth training or depth-checkpoint mutation.
- Effective split after full class audit: Scene02/06/18 train, Scene01
  validation, Scene20 untouched final test; all ten variations follow their
  scene. GuardRail is found only in Scene20, so the scene-disjoint trainable
  taxonomy is thirteen classes with GuardRail ignored.
- Fourteen VKITTI classes; `Undefined` maps to ignore ID 255. Verify actual colors and class support before training.
- Preserve YOLO model layers 0–6 exactly, including BatchNorm buffers. Use the same A10 GPU as the earlier full SceneFlow run.
- Use `uv` project management (`modal.Image.uv_sync`), not `uv pip` or a standalone requirements install. Use wheels.
- Submit full training detached so laptop connectivity is unnecessary. No existing runs may be resumed.
- The worktree contains other user edits; do not commit or rewrite them.

## Review Focus

- A mask has an unknown color: converter raises with the color and source path (Task 1 test).
- A scene or variation leaks into two splits: manifest builder rejects it (Task 1 test).
- A frozen parameter *or BatchNorm buffer* changes: trainer fails before accepting a checkpoint (Task 2 test).
- A class is absent from validation: audit reports support and mIoU interpretation, rather than silently treating it as good performance (Task 1 test).
- A remote run restarts: checkpoints and split manifest remain reproducible on the output Volume (Task 3 test/probe).

---

### Task 1: VKITTI semantic data preparation

**Files:** Create `experiments/vkitti2/semantic_data.py`; test in `tests/test_vkitti2_semantic_data.py`.

**Interfaces:** `decode_class_mask(rgb: np.ndarray, source: str) -> np.ndarray` returns uint8 class IDs; `build_split_index(members: Iterable[str]) -> dict[str, list[tuple[str, str]]]` groups image/mask archive members by the fixed scene split; `prepare_dataset(rgb_tar: Path, mask_tar: Path, destination: Path) -> dict` writes mirrored `images/{split}` and `masks/{split}` plus audit JSON/YAML. Convert all left-camera RGB/class images; fail on missing pairs or unexpected colors.

- [ ] Write tests for 14 colors, Undefined ignore, unknown color, pairing, split isolation, and class-support reporting.
- [ ] Run `uv run pytest tests/test_vkitti2_semantic_data.py -q` and confirm missing-interface failures.
- [ ] Implement stream extraction/conversion and dataset YAML; avoid extracting the full RGB tar to the Volume.
- [ ] Run the tests and inspect a few converted masks at original resolution.

### Task 2: Frozen semantic training and verification

**Files:** Create `experiments/vkitti2/semantic_train.py`; test in `tests/test_vkitti2_semantic_train.py`.

**Interfaces:** `assert_trunk_unchanged(source, trained, end_layer=6) -> None` checks parameters and buffers; `train_semantic(dataset_yaml: Path, source_weights: Path, out_dir: Path, *, probe: bool, ...) -> dict` trains Ultralytics semantic with `freeze=7`, A10-probed batch, AdamW, near-native resolution, scene-held-out validation, checkpointing and per-class metrics. Verify source/target preprocessing and the source checkpoint hash.

- [ ] Write tests for trunk equality, changed weight, changed BN buffer, and configuration (`freeze=7`, task semantic, correct source model).
- [ ] Confirm expected failures with `uv run pytest tests/test_vkitti2_semantic_train.py -q`.
- [ ] Implement training wrapper and post-training trunk verification; write run metadata with versions, args, class names, split hashes, and checkpoints.
- [ ] Run focused tests and `uv run python -m py_compile` on created files.

### Task 3: Modal A10 probe and detached run

**Files:** Create `experiments/vkitti2/modal_semantic.py`; document in `experiments/vkitti2/SEMANTIC_MODAL_RUN.md`.

**Interfaces:** CPU `prepare_full` consumes compressed archives and builds container-local data; GPU `probe_a10` validates first batches, memory and throughput; GPU `train_a10` trains full data, persists to `svde-results`, and can resume. `launch_a10` submits the training call and exits under `uv run modal run -d`.

- [ ] Run local static checks and Modal source-volume listing.
- [ ] Run CPU preparation and inspect counts, class support, mask sizes, YAML, and a few images.
- [ ] Run a short A10 probe, ensure layers 0–6 unchanged and memory/throughput acceptable.
- [ ] If probe passes, submit full training detached; record app/call IDs, exact commands, dataset audit, cost estimate, and monitoring commands.
- [ ] Verify the Modal app is active and first checkpoint/metrics appear before reporting launch.

# Toward Generalizable Cigarette Detection in Surveillance

Code and the SHA-locked detector for **"Toward Generalizable Cigarette Detection in Surveillance:
Tile-Based Augmentation, Hard-Negative Mining, and Temporal Stabilization"** (under review).

The paper studies a single-class cigarette detector for surveillance video and evaluates three
components separately, each for *generalization* rather than single-setting accuracy:

| Component | What it does | Measured effect |
|---|---|---|
| Gated 2×2 tiling at inference | Splits the frame into four overlapping half-resolution tiles when the full-frame pass returns nothing | Recall 0.70 → 0.94 at a precision cost of 0.83 → 0.62; F1 essentially unchanged |
| Hard-negative look-alike data | Trains on deliberately collected pens, markers and cups held at the mouth | Removing them roughly doubles the per-frame false-positive rate on held-out subjects (0.071 → 0.140) |
| Temporal event stabilization | Causal finite-state machine with asymmetric hysteresis over the per-frame presence signal | About 98% of per-minute false alarms removed |

Generalization is reported rather than assumed: a cross-source gap (AP@50 0.873 on the public
benchmark vs 0.565 on held-out in-house footage), corruption robustness, calibration, and a
within-site placement stress test.

---

## Released detector

`weights/yolo11m_cigarette_finetuned.pt` — YOLO11m, single class (`cigarette`), fine-tuned on a
377K-image merged dataset.

```
SHA-256  17BF89F3BBDCC2B85315627485B1EE56CF4B67C826B114EE6EB3A36C583E3D14
```

This is the exact checkpoint behind every number in the paper. Verify before use:

```powershell
certutil -hashfile weights\yolo11m_cigarette_finetuned.pt SHA256
```

Deployment operating point: confidence **0.30**, matching IoU **0.30**, `imgsz` 1280 with a 1.25×
bicubic upscale; the 2×2 tile pass uses 20% overlap and fires only when the full-frame pass returns
no boxes.

---

## Setup

Python 3.10+ is required. PyTorch is deliberately **not pinned** in `requirements.txt`; install the
build that matches your CUDA first.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The development machine is an RTX 5060 Ti (Blackwell, sm_120), which needs the CUDA 12.8 build or
Ultralytics fails with "no kernel image is available".

---

## Scripts and the sections they produce

| Script | Section |
|---|---|
| `scripts/eval_video_metrics.py` | Video metrics engine: prediction, frame-level PR, event-level FSM metrics (§5.2, §5.3, Tables 2–3) |
| `scripts/fsm_param_sweep.py` | Confirmation window, `K_off` and `N_min` sweep (§5.3) |
| `scripts/ab5_notile_ablation.py` | Inference-time tiling ablation with the deployment gate (§5.6d) |
| `scripts/ab5_build_traintime.py` | Builds the training-time tiling arm, 2×2 tiles with label remapping (§5.5) |
| `scripts/ab5_diagnostic.py` | Scale-matching diagnostic for the training-time grid (§5.5) |
| `scripts/bench_tiling_walltime.py` | GPU wall-clock for full-frame and tiled passes (§6) |
| `scripts/complexity_analysis.py` | Complexity analysis of tiling, tile-NMS and the temporal stage (§6) |
| `scripts/placement_predict_boxes.py` | Runs the locked detector over the placement recordings (§5.6) |
| `scripts/placement_eval.py` | Per-position and per-scenario placement metrics (§5.6, Table 4) |
| `scripts/e42_compare_day_evening.py` | Daylight vs artificial-light comparison of the same grid (§5.6) |
| `scripts/e42_scenario_sweep.py` | Confidence sweep per scenario across both sessions (§5.6) |
| `scripts/e41_detector_eval.py` | Detector evaluation on a subject absent from training, with the size-matched control (§5.6) |
| `scripts/optuna_tune.py` | Optuna search behind the architecture screening (§5.1) |

The example below reproduces the event-level numbers once you have a video and its ground truth:

```powershell
python scripts/eval_video_metrics.py predict --source yolo `
    --video <video.mp4> --weights weights/yolo11m_cigarette_finetuned.pt --out pred.jsonl

python scripts/eval_video_metrics.py eval --video <video.mp4> --pred pred.jsonl `
    --gt_csv <gt.csv> --out_dir out --conf_thr 0.30 --iou_thr 0.30 `
    --event_confirm 60 --event_stop_miss 8 --event_min_dur_s 0.8
```

`--event_confirm` is in frames: the paper's 2 s confirmation window is 60 frames at the 30 FPS of
the recordings.

> **Paths.** These scripts were written for one machine and several still carry an absolute
> `ROOT` constant at the top. Edit it before running, or run from a checkout laid out the same way.

---

## Data

**Public benchmark.** The training set merges four openly available cigarette-detection datasets
from Roboflow Universe; the merged, re-released version is linked from the paper. The external
head-to-head baseline is the Hugging Face release
[`Enos-123/smoking-detection`](https://huggingface.co/Enos-123/smoking-detection), evaluated
unmodified.

**In-house footage is not included and will not be released.** The deployment video, the placement
recordings and the look-alike clips are human-subject recordings collected under informed consent
and an ethics approval that does not permit redistribution, and under the Turkish Personal Data
Protection Law (KVKK). No image, video or frame of any participant appears in this repository.
De-identified derived data (labels and per-frame detection records used in the evaluations) is
available from the corresponding author on reasonable request.

---

## License

Released under **AGPL-3.0**, matching the license of the Ultralytics YOLO11 framework the detector
is fine-tuned from. If you need different terms for the weights, Ultralytics licensing applies to
the underlying model.

---

## Citation

A citation entry will be added once the paper is published. Until then please cite it as under
review.

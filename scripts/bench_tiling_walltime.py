# -*- coding: utf-8 -*-
r"""OPSIYONEL: SAHI tiling GPU wall-clock benchmark (Paper 1 fig15 dogrulamasi).
KENDI TERMINALINDE calistir (GPU + model gerekir). ~1-2 dk, tqdm ile canli ilerleme.

CAVEAT: Bu olcum Theta(T^2) BOUND'unu TEST ETMEZ. Kesin iddia = pass-count/FLOPs (COMPLEXITY_ANALYSIS.md
Bolum 1). GPU'da per-call CUDA/Python launch overhead + kucuk tile'larda dusuk doluluk yuzunden
batched tile'lar cogu zaman SUB-quadratic olceklenir -> bu bir realtime KAZANCI, celiski degil.

Kullanim (proje kokunden):
  py -3.10 .\tools\bench_tiling_walltime.py
  py -3.10 .\tools\bench_tiling_walltime.py --weights models\yolo11m_full_dataset_finetune_e38_best.pt --imgsz 960 --reps 20
"""
import os
os.environ["PYTHONUTF8"] = "1"
import argparse
import time
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_W = os.path.join(ROOT, "models", "yolo11m_full_dataset_finetune_e38_best.pt")
FIG = os.path.join(ROOT, "runs_v3_eval", "figures")
P1 = os.path.join(ROOT, "PAPERS", "PAPER1_detector_tiling_temporal")


def tile_windows(w, h, gx, gy, overlap):
    tw = int(np.ceil(w / gx)); th = int(np.ceil(h / gy))
    sx = int(max(1, tw * (1.0 - overlap))); sy = int(max(1, th * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tw + 1), sx)) or [0]
    ys = list(range(0, max(1, h - th + 1), sy)) or [0]
    wins = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(w, x1 + tw); y2 = min(h, y1 + th)
            wins.append((max(0, x2 - tw), max(0, y2 - th), x2, y2))
    return wins


def predict(model, frame, imgsz, conf, iou):
    model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)


def bench_grid(model, frame, T, imgsz, conf, iou, reps, overlap):
    h, w = frame.shape[:2]
    wins = tile_windows(w, h, T, T, overlap) if T > 1 else [(0, 0, w, h)]
    # warmup
    for _ in range(3):
        for (x1, y1, x2, y2) in wins:
            predict(model, frame[y1:y2, x1:x2], imgsz, conf, iou)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for (x1, y1, x2, y2) in wins:
            predict(model, frame[y1:y2, x1:x2], imgsz, conf, iou)
        times.append(time.perf_counter() - t0)
    return len(wins), float(np.median(times)), float(np.percentile(times, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEF_W)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--overlap", type=float, default=0.22)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--grids", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--frame", default="", help="opsiyonel gercek kare; yoksa sentetik 1920x1080")
    args = ap.parse_args()

    print(f"[bench] model={args.weights} imgsz={args.imgsz} reps={args.reps}", flush=True)
    model = YOLO(args.weights)
    if args.frame and os.path.exists(args.frame):
        frame = cv2.imdecode(np.fromfile(args.frame, np.uint8), cv2.IMREAD_COLOR)
    else:
        frame = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
    print(f"[bench] frame={frame.shape[1]}x{frame.shape[0]}", flush=True)

    rows = ["T,n_tiles,median_ms,p10_ms,per_tile_ms"]
    Ts, mids, nts = [], [], []
    for T in tqdm(args.grids, desc="grids"):
        nt, med, p10 = bench_grid(model, frame, T, args.imgsz, args.conf, args.iou, args.reps, args.overlap)
        rows.append(f"{T},{nt},{med*1e3:.2f},{p10*1e3:.2f},{med*1e3/nt:.2f}")
        Ts.append(T); mids.append(med * 1e3); nts.append(nt)
        tqdm.write(f"  T={T} tiles={nt}  median={med*1e3:.1f}ms  per-tile={med*1e3/nt:.1f}ms")

    out_csv = os.path.join(ROOT, "runs_v3_eval", "tiling_walltime.csv")
    open(out_csv, "w", encoding="utf-8").write("\n".join(rows) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(Ts, mids, "-o", color="#c0392b", lw=2, label="measured median (GPU)")
    ref = mids[0] * np.array(nts) / nts[0]
    ax.plot(Ts, ref, "--", color="gray", lw=1.5, label="$\\propto N_t$ (linear-in-passes)")
    ax.set_xlabel("Tile grid factor $T$"); ax.set_ylabel("Tiled inference wall-clock (ms/frame)")
    ax.set_title("Tiling GPU wall-clock: measured per-tile latency vs. tile count")
    ax.set_xticks(Ts); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    for d in (FIG, os.path.join(P1, "figures")):
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, "fig15b_tiling_walltime.png"), dpi=300)
    plt.close()
    print(f"\nSAVED: {out_csv} + fig15b_tiling_walltime.png", flush=True)


if __name__ == "__main__":
    main()

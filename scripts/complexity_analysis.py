# -*- coding: utf-8 -*-
"""PAPER 1 - Computational complexity analysis (theory + empirical validation).
Uc bilesen: (1) SAHI tiling (kesin pass-count/FLOPs), (2) tile-NMS (worst/typical, log-log fit),
(3) temporal build_events (Theta(N), online). Figur (fig15/16/17) + complexity_summary.csv + rapor.
GPU wall-clock AYRI script'te (kullanicinin terminaline) -- burasi GPU'suz, saniyeler surer."""
import os
os.environ["PYTHONUTF8"] = "1"
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # Elsevier: TrueType (Type-3 reddedilir)
import matplotlib.pyplot as plt

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
EV = os.path.join(ROOT, "runs_v3_eval")
FIG = os.path.join(EV, "figures")
P1 = os.path.join(ROOT, "PAPERS", "PAPER1_detector_tiling_temporal")
os.makedirs(FIG, exist_ok=True)
os.makedirs(os.path.join(P1, "figures"), exist_ok=True)
os.makedirs(os.path.join(P1, "tables"), exist_ok=True)


# ---- apiserver._tile_windows'un birebir kopyasi (tile sayimi icin) ----
def tile_windows(w, h, gx, gy, overlap):
    gx = max(1, int(gx)); gy = max(1, int(gy))
    overlap = float(max(0.0, min(0.6, overlap)))
    tw = int(np.ceil(w / gx)); th = int(np.ceil(h / gy))
    sx = int(max(1, tw * (1.0 - overlap))); sy = int(max(1, th * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tw + 1), sx)) or [0]
    ys = list(range(0, max(1, h - th + 1), sy)) or [0]
    return len(xs) * len(ys)


# ---- eval_video_metrics.build_events_from_binary'nin birebir kopyasi ----
def build_events_from_binary(present, fps, confirm_frames, stop_miss_frames, min_dur_s):
    confirm_frames = max(1, int(confirm_frames)); stop_miss_frames = max(1, int(stop_miss_frames))
    min_frames = max(1, int(round(min_dur_s * fps)))
    events = []; in_ev = False; on_count = 0; off_count = 0; start_frame = 0
    for i in range(len(present)):
        if present[i] == 1:
            on_count += 1; off_count = 0
            if (not in_ev) and (on_count >= confirm_frames):
                in_ev = True; start_frame = i - confirm_frames + 1
        else:
            off_count += 1; on_count = 0
            if in_ev and (off_count >= stop_miss_frames):
                end_frame = i - stop_miss_frames
                if end_frame >= start_frame and (end_frame - start_frame + 1) >= min_frames:
                    events.append((start_frame, end_frame))
                in_ev = False; off_count = 0
    if in_ev:
        end_frame = len(present) - 1
        if end_frame >= start_frame and (end_frame - start_frame + 1) >= min_frames:
            events.append((start_frame, end_frame))
    return events


def nms_boxes(boxes_xywh, scores, iou_thr=0.5):
    """_nms_xyxy'nin cagirdigi cekirdek: cv2.dnn.NMSBoxes (greedy)."""
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=float(iou_thr))
    return len(idxs) if idxs is not None else 0


# ============ 1) TILING: kesin pass-count / FLOPs (GPU'suz, teori) ============
def fig_tiling():
    W, H = 1920, 1080; OV = 0.22
    Ts = [1, 2, 3, 4, 5]
    Nt = [tile_windows(W, H, t, t, OV) for t in Ts]        # gercek tile sayisi (overlap dahil)
    T2 = [t * t for t in Ts]                               # nominal T^2
    depl_worst = [1 + n for n in Nt]                       # deployed worst-case: full + tiles
    # rapor: T=2 -> Nt=4 (birebir _tile_windows ile), 1+4=5 deployed worst
    print(f"[tiling] W={W}xH={H} ov={OV} | T={Ts} Nt={Nt} 1+Nt={depl_worst}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    # panel A: forward-pass sayisi
    ax[0].plot(Ts, Nt, "-o", color="#c0392b", lw=2, label="Ablation (tiles only) = $N_t$")
    ax[0].plot(Ts, depl_worst, "-s", color="#2b7bba", lw=2, label="Deployed worst = $1+N_t$")
    ax[0].plot(Ts, T2, "--", color="gray", lw=1.5, label="$T^2$ reference")
    ax[0].set_xlabel("Tile grid factor $T$ ($T{\\times}T$)"); ax[0].set_ylabel("Forward passes / frame")
    ax[0].set_title("Tiling forward-pass count $=\\Theta(T^2)$"); ax[0].set_xticks(Ts)
    ax[0].legend(); ax[0].grid(alpha=.3)
    for t, n in zip(Ts, Nt):
        ax[0].annotate(str(n), (t, n), textcoords="offset points", xytext=(4, 6), fontsize=8)
    # panel B: goreli detektor FLOPs -- tiling (Nt * imgsz^2) vs esdeger tek-kare (T^2 * imgsz^2)
    tiling_flops = np.array(Nt, float)                     # / imgsz^2 sabiti
    fullframe_equiv = np.array(T2, float)
    ax[1].plot(Ts, tiling_flops, "-o", color="#c0392b", lw=2, label="Tiling: $N_t\\,{\\cdot}\\,s^2$")
    ax[1].plot(Ts, fullframe_equiv, "-^", color="#27ae60", lw=2,
               label="Full-frame @ $T{\\cdot}s$: $T^2 s^2$")
    ax[1].set_xlabel("Tile grid factor $T$"); ax[1].set_ylabel("Detector FLOPs / $s^2$ (relative)")
    ax[1].set_title("Tiling cost $\\Theta(T^2 s^2)$, matching equal-resolution\nup to the 22% tile-overlap factor (exact at deployed $T{=}2$: $N_t{=}4$)")
    ax[1].set_xticks(Ts); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    for d in (FIG, os.path.join(P1, "figures")):
        plt.savefig(os.path.join(d, "fig15_complexity_tiling.png"), dpi=300)
        plt.savefig(os.path.join(d, "fig15_complexity_tiling.pdf"), dpi=300)
    plt.close()
    return Ts, Nt, depl_worst, T2


# ============ 2) NMS: worst-case (all-kept) vs typical, log-log fit ============
def make_nonoverlap(n):
    """Ortusmeyen kutular -> hicbiri baskilanmaz -> hepsi kalir -> O(n^2) worst-case."""
    g = int(np.ceil(np.sqrt(n))); step = 40; sz = 20
    b = []
    for k in range(n):
        r, c = divmod(k, g)
        b.append([c * step, r * step, sz, sz])
    return b


def make_clustered(n):
    """Az sayida merkez etrafinda yigin -> cogu baskilanir -> ~O(n log n) tipik."""
    rng = np.random.default_rng(0); centers = rng.integers(0, 2000, size=(8, 2))
    b = []
    for k in range(n):
        cx, cy = centers[k % len(centers)]
        jx, jy = rng.integers(-10, 10, size=2)
        b.append([int(cx + jx), int(cy + jy), 60, 60])
    return b


def time_nms(builder, ns, reps=7):
    ts = []
    for n in ns:
        boxes = builder(n); scores = list(np.linspace(1.0, 0.1, n))
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter(); nms_boxes(boxes, scores, 0.5); dt = time.perf_counter() - t0
            best = min(best, dt)
        ts.append(best)
    return np.array(ts)


def fig_nms():
    ns = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    t_worst = time_nms(make_nonoverlap, ns)
    t_typ = time_nms(make_clustered, ns)
    # log-log egim fiti (buyuk-n bolgesi, n>=128)
    m = ns >= 128
    sw = np.polyfit(np.log(ns[m]), np.log(t_worst[m]), 1)[0]
    st = np.polyfit(np.log(ns[m]), np.log(t_typ[m]), 1)[0]
    print(f"[nms] fitted slope worst={sw:.2f} (theory 2)  typical={st:.2f} (theory ~1)", flush=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.loglog(ns, t_worst * 1e3, "-o", color="#c0392b", label=f"Worst (all-kept), slope={sw:.2f}")
    ax.loglog(ns, t_typ * 1e3, "-s", color="#2b7bba", label=f"Typical (clustered), slope={st:.2f}")
    # referans egriler
    ref2 = t_worst[m][0] * (ns / ns[m][0]) ** 2
    ref1 = t_typ[m][0] * (ns / ns[m][0]) ** 1
    ax.loglog(ns, ref2 * 1e3, "--", color="gray", lw=1, label="$O(n^2)$")
    ax.loglog(ns, ref1 * 1e3, ":", color="gray", lw=1, label="$O(n)$")
    ax.axvspan(8, 20, color="green", alpha=0.10)
    ax.annotate("deployment\n$n\\approx$8-20", (12, ax.get_ylim()[0] * 3), fontsize=8, color="green", ha="center")
    ax.set_xlabel("Merged boxes $n$  (log scale)"); ax.set_ylabel("NMS time (ms)  (log scale)")
    ax.set_title("Tile-NMS scaling (log-log axes): $\\Theta(n\\log n)$ typical, $O(n^2)$ worst\n(at deployment $n$ is small, negligible)")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    plt.tight_layout()
    for d in (FIG, os.path.join(P1, "figures")):
        plt.savefig(os.path.join(d, "fig16_complexity_nms.png"), dpi=300)
        plt.savefig(os.path.join(d, "fig16_complexity_nms.pdf"), dpi=300)
    plt.close()
    return ns, t_worst, t_typ, sw, st


# ============ 3) TEMPORAL: build_events Theta(N), online O(1) memory ============
def fig_temporal():
    Ns = [1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000]
    rng = np.random.default_rng(0); ts = []
    for N in Ns:
        present = (rng.random(N) < 0.3).astype(np.int8)
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            build_events_from_binary(present, 25.0, 3, 8, 0.8)
            best = min(best, time.perf_counter() - t0)
        ts.append(best); print(f"[temporal] N={N:>8}  t={best*1e3:8.2f} ms", flush=True)
    ts = np.array(ts); Ns = np.array(Ns)
    slope = np.polyfit(np.log(Ns), np.log(ts), 1)[0]
    thr = Ns / ts  # frames/sec throughput
    print(f"[temporal] log-log slope={slope:.3f} (theory 1.0) | throughput~{thr.mean():.2e} frame/s", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(Ns / 1e6, ts * 1e3, "-o", color="#27ae60", lw=2)
    ax[0].set_xlabel("Frames $N$ (millions)"); ax[0].set_ylabel("build_events time (ms)")
    ax[0].set_title(f"Temporal stabilization is linear $\\Theta(N)$\n(linear axes; slope {slope:.2f} from log-log fit, right)"); ax[0].grid(alpha=.3)
    ax[1].loglog(Ns, ts * 1e3, "-o", color="#27ae60", lw=2, label="measured")
    ref = ts[0] * (Ns / Ns[0])
    ax[1].loglog(Ns, ref * 1e3, "--", color="gray", label="$\\Theta(N)$ slope 1")
    ax[1].set_xlabel("Frames $N$  (log scale)"); ax[1].set_ylabel("time (ms)  (log scale)")
    ax[1].set_title("O(1) memory / online — log-log (slope≈1, linear)"); ax[1].legend(); ax[1].grid(alpha=.3, which="both")
    plt.tight_layout()
    for d in (FIG, os.path.join(P1, "figures")):
        plt.savefig(os.path.join(d, "fig17_complexity_temporal.png"), dpi=300)
        plt.savefig(os.path.join(d, "fig17_complexity_temporal.pdf"), dpi=300)
    plt.close()
    return Ns, ts, slope, thr.mean()


def main():
    print("=== PAPER 1 COMPLEXITY ANALYSIS (theory + CPU empirical) ===", flush=True)
    Ts, Nt, depl, T2 = fig_tiling()
    ns, tw, tt, sw, st = fig_nms()
    Ns, tt_temp, slope, thr = fig_temporal()

    # ---- ozet tablo CSV ----
    rows = [
        "component,best_case,average_case,worst_case,space,online_streaming,dominant_term,deployment_typical",
        "SAHI_tiling_ablation,Theta(T^2 s^2),Theta(T^2 s^2),Theta(T^2 s^2),Theta(s^2) per tile,no (batchable),T^2 forward passes,"
        "T=2 -> 4 passes (Nt=4)",
        "SAHI_tiling_deployed,Theta(s^2) [gated skip],~Theta(s^2) [amortized],Theta((1+T^2) s^2),Theta(s^2),no,gated 2nd pass,"
        "1 pass typ / 5 worst",
        "tile_NMS,O(n),Theta(n log n),O(n^2),O(n),n/a,greedy suppression,n<20 -> negligible",
        "temporal_build_events,Theta(N),Theta(N),Theta(N),O(1) work + O(E) out,yes (causal),single pass,"
        f"{thr:.2e} frame/s",
    ]
    csvtxt = "\n".join(rows) + "\n"
    for d in (EV, os.path.join(P1, "tables")):
        open(os.path.join(d, "complexity_summary.csv"), "w", encoding="utf-8").write(csvtxt)

    # ---- ampirik olcum CSV (dogrulama) ----
    emp = ["measurement,x,value"]
    for t, n in zip(Ts, Nt):
        emp.append(f"tiling_Nt,{t},{n}")
    for n, a, b in zip(ns, tw, tt):
        emp.append(f"nms_worst_ms,{n},{a*1e3:.4f}"); emp.append(f"nms_typ_ms,{n},{b*1e3:.4f}")
    for n, t in zip(Ns, tt_temp):
        emp.append(f"temporal_ms,{n},{t*1e3:.4f}")
    emp.append(f"nms_slope_worst,fit,{sw:.3f}"); emp.append(f"nms_slope_typ,fit,{st:.3f}")
    emp.append(f"temporal_slope,fit,{slope:.3f}")
    for d in (EV, os.path.join(P1, "tables")):
        open(os.path.join(d, "complexity_empirical.csv"), "w", encoding="utf-8").write("\n".join(emp) + "\n")

    print("\nSAVED: fig15/16/17 + complexity_summary.csv + complexity_empirical.csv", flush=True)
    print(f"  nms slope worst={sw:.2f}/typ={st:.2f} | temporal slope={slope:.3f} | throughput={thr:.2e} f/s", flush=True)


if __name__ == "__main__":
    main()

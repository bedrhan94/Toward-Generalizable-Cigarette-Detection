# -*- coding: utf-8 -*-
r"""AB5 — NO-TILE vs TILE ablation (Paper 1, tiling katkisi).

NE OLCER:
  Ayni yuksek-cozunurluklu etiketli karelerde dedektoru IKI modda kosturur:
    * no-tile : yalniz full-frame gecis (imgsz, upscale AYNI).
    * tile    : DAGITIM-SADIK gated 2x2 -> full-frame; 0 kutu ise 2x2 tile ikinci gecis.
  Boylece fark = SADECE tiling'in katkisi (ayni kareler, ayni upscale).

DANISMAN KURALLARI (uygulandi):
  1) Manşet = gated operating-point (dagitimda gercekten olan). Caption: "tiling yalniz
     full-frame conf>=CONF'ta 0 tespit verince atesler." Tile-always/AP50 = ikincil tavan.
  2) Gate ΔmAP'i seyreltir -> n_gate_fired + gate-atesli alt-kumede RECOVERED-FN raporlanir.
  3) `_secondpass_allowed` wall-clock throttle YOK (offline). no-tile upscale = tile ile AYNI.
  4) Negatiflerde tiling FP ekler -> ΔPrecision de raporlanir (tiling'in maliyeti).
  Leakage: iki mod AYNI karelerde -> fark adil; mutlak mAP overlap varsa iyimser.
  Runner per-frame kaynak listesi yazar (sonra Full_Dataset ile overlap kontrolu icin).

KOSMA (veri geldiginde):
  .\.venv\Scripts\python.exe tools\ab5_notile_ablation.py ^
     --images <notile>/images --labels <notile>/labels ^
     --weights runs/final/01_yolo11m_siha_optuna_final_e80/01_train_run/weights/best.pt ^
     --out runs_v3_eval/AB5_notile
  # smoke: --limit 40
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ======================================================= tiling (apiserver ile BIREBIR)
def _tile_windows(w, h, gx, gy, overlap):
    gx = max(1, int(gx)); gy = max(1, int(gy))
    overlap = float(max(0.0, min(0.6, overlap)))
    tw = int(np.ceil(w / gx)); th = int(np.ceil(h / gy))
    sx = int(max(1, tw * (1.0 - overlap))); sy = int(max(1, th * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tw + 1), sx)) or [0]
    ys = list(range(0, max(1, h - th + 1), sy)) or [0]
    wins = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(w, x1 + tw); y2 = min(h, y1 + th)
            x1 = max(0, x2 - tw); y1 = max(0, y2 - th)
            wins.append((x1, y1, x2, y2))
    return wins


def _nms_xyxy(boxes, confs, iou_thr):
    if not boxes:
        return [], []
    b_xywh = [[int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))] for (x1, y1, x2, y2) in boxes]
    scores = [float(c) for c in confs]
    idxs = cv2.dnn.NMSBoxes(b_xywh, scores, score_threshold=0.0, nms_threshold=float(iou_thr))
    if idxs is None or len(idxs) == 0:
        return [], []
    idxs = idxs.flatten().tolist()
    return [boxes[i] for i in idxs], [scores[i] for i in idxs]


def _scale_frame(frame, upscale):
    if abs(upscale - 1.0) < 1e-6:
        return frame, 1.0
    h, w = frame.shape[:2]
    interp = cv2.INTER_CUBIC if upscale > 1.0 else cv2.INTER_AREA
    return cv2.resize(frame, (int(w * upscale), int(h * upscale)), interpolation=interp), upscale


def _unscale(box, sc, w0, h0):
    x1, y1, x2, y2 = box
    if abs(sc - 1.0) < 1e-6:
        return (int(x1), int(y1), int(x2), int(y2))
    x1, y1, x2, y2 = x1 / sc, y1 / sc, x2 / sc, y2 / sc
    return (max(0, min(w0 - 1, int(round(x1)))), max(0, min(h0 - 1, int(round(y1)))),
            max(0, min(w0, int(round(x2)))), max(0, min(h0, int(round(y2)))))


class Detector:
    def __init__(self, weights, imgsz, upscale, iou, max_det=80, tile_grid=(2, 2), tile_ov=0.22):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.imgsz = int(imgsz); self.upscale = float(upscale); self.iou = float(iou)
        self.max_det = int(max_det); self.grid = tile_grid; self.ov = float(tile_ov)

    def _predict(self, frame, conf, imgsz=None, max_det=None):
        r = self.m.predict(frame, conf=float(conf), iou=self.iou,
                           imgsz=int(imgsz or self.imgsz), max_det=int(max_det or self.max_det),
                           verbose=False)[0]
        boxes, confs = [], []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                try:
                    confs.append(float(b.conf[0]))
                except Exception:
                    confs.append(0.5)
        return boxes, confs

    def full_frame(self, frame, conf):
        """no-tile ilk gecis (upscale UYGULANIR — tile ile AYNI)."""
        yin, sc = _scale_frame(frame, self.upscale)
        bs, cs = self._predict(yin, conf, self.imgsz)
        h0, w0 = frame.shape[:2]
        return [_unscale(b, sc, w0, h0) for b in bs], cs

    def tiled(self, frame, conf):
        """2x2 tile (upscale UYGULANMAZ — apiserver de tile'da ham kareyi bolar)."""
        h, w = frame.shape[:2]
        allb, allc = [], []
        for (x1, y1, x2, y2) in _tile_windows(w, h, self.grid[0], self.grid[1], self.ov):
            t = frame[y1:y2, x1:x2]
            if t.size == 0:
                continue
            tb, tc = self._predict(t, conf, self.imgsz, max_det=max(20, self.max_det // 2))
            for (bx1, by1, bx2, by2), bc in zip(tb, tc):
                allb.append((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)); allc.append(bc)
        return _nms_xyxy(allb, allc, self.iou)


# ======================================================= GT + metrik
def load_gt(label_path, w, h):
    """YOLO norm xywh -> abs xyxy listesi. label_path None/yok ise NEGATIF (bos GT)."""
    out = []
    if not label_path or not os.path.exists(label_path):
        return out
    with open(label_path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            p = ln.split()
            if len(p) < 5:
                continue
            try:
                cx, cy, bw, bh = float(p[1]) * w, float(p[2]) * h, float(p[3]) * w, float(p[4]) * h
            except ValueError:
                continue
            out.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr=0.5):
    """preds: [(box,conf)] conf DESC sirali. Doner: tp_flags (pred sirasinda), matched_gt_idx set."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    used = set(); tp = [0] * len(preds)
    for i in order:
        best, bj = thr, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(preds[i][0], g)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj); tp[i] = 1
    return tp, used


def voc_ap(recs, precs):
    mrec = [0.0] + list(recs) + [1.0]; mpre = [0.0] + list(precs) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def size_bin(box):
    a = (box[2] - box[0]) * (box[3] - box[1])
    s = a ** 0.5
    return "small (<32 px)" if s < 32 else ("medium (32-96)" if s < 96 else "large (>96)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="goruntu klasoru (recursive taranir)")
    ap.add_argument("--labels", required=True, help="YOLO label klasoru (recursive)")
    ap.add_argument("--weights", default="runs/final/01_yolo11m_siha_optuna_final_e80/01_train_run/weights/best.pt")
    ap.add_argument("--out", default="runs_v3_eval/AB5_notile")
    ap.add_argument("--conf", type=float, default=0.16, help="dagitim operating-point (cam1)")
    ap.add_argument("--second-conf", type=float, default=0.12, help="tile ikinci gecis conf (cam1)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--upscale", type=float, default=1.25)
    ap.add_argument("--iou", type=float, default=0.45, help="NMS/dedektor iou")
    ap.add_argument("--lowconf", type=float, default=0.001, help="AP50 egrisi icin")
    ap.add_argument("--match-iou", type=float, default=0.5, help="TP eslesme IoU")
    ap.add_argument("--limit", type=int, default=0, help="smoke: ilk N kare")
    args = ap.parse_args()

    # ---- veri: stem -> (img, lbl) ----
    imgs = {}
    for root, _, files in os.walk(args.images):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                imgs.setdefault(os.path.splitext(f)[0], os.path.join(root, f))
    lbls = {}
    for root, _, files in os.walk(args.labels):
        for f in files:
            if f.lower().endswith(".txt"):
                lbls.setdefault(os.path.splitext(f)[0], os.path.join(root, f))
    # TUM goruntuler islenir; label dosyasi OLMAYAN goruntu = NEGATIF (bos GT).
    # (tiling'in negatiflerde FP maliyetini = ΔPrecision olcmek icin sart — danisman.)
    stems = sorted(imgs)
    if args.limit:
        stems = stems[::max(1, len(stems) // args.limit)][:args.limit]   # temsili altornek (blok degil)
    if not stems:
        print("!! goruntu YOK"); return
    n_pos = sum(1 for s in stems if s in lbls)
    print(f"[AB5] {len(stems)} kare ({n_pos} pozitif + {len(stems)-n_pos} negatif) | "
          f"weights={os.path.basename(args.weights)}")

    det = Detector(args.weights, args.imgsz, args.upscale, args.iou)

    # toplayicilar
    modes_op = ["no_tile", "tile_gated"]
    tp_op = {m: 0 for m in modes_op}; fp_op = {m: 0 for m in modes_op}
    ap_preds = {"no_tile": [], "tile_always": [], "tile_only": []}   # (conf, tp) global — AP icin
    total_gt = 0
    size_gt = {}; size_hit = {m: {} for m in modes_op}   # boyut-katmanli recall
    n_gate_fired = 0; recovered_fn = 0; gate_notile_miss = 0
    n_neg = 0; tiling_fp_on_neg = 0
    per_frame = []

    t0 = time.time()
    for k, s in enumerate(stems):
        img = cv2.imdecode(np.fromfile(imgs[s], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt(lbls.get(s), w, h)   # label yoksa NEGATIF (bos GT)
        total_gt += len(gts)
        for g in gts:
            size_gt[size_bin(g)] = size_gt.get(size_bin(g), 0) + 1
        is_neg = (len(gts) == 0)
        if is_neg:
            n_neg += 1

        # --- LOW-CONF predictions (AP egrisi + operating-point turetme) ---
        ff_lb, ff_lc = det.full_frame(img, args.lowconf)          # full-frame low
        tl_lb, tl_lc = det.tiled(img, args.lowconf)               # tile low
        # tile_always = full + tile merge (AP tavani)
        ta_b, ta_c = _nms_xyxy(ff_lb + tl_lb, ff_lc + tl_lc, args.iou)

        # --- operating-point ---
        # no_tile @ conf
        nt_op = [(b, c) for b, c in zip(ff_lb, ff_lc) if c >= args.conf]
        # tile_gated: no_tile bos ise tile @ second_conf, degilse no_tile
        gate = (len(nt_op) == 0)
        if gate:
            n_gate_fired += 1
            tg_op = [(b, c) for b, c in zip(tl_lb, tl_lc) if c >= args.second_conf]
        else:
            tg_op = nt_op

        # --- metrik: operating-point TP/FP + boyut recall ---
        for mode, preds in (("no_tile", nt_op), ("tile_gated", tg_op)):
            tp, used = match(preds, gts, args.match_iou)
            tp_op[mode] += sum(tp); fp_op[mode] += (len(preds) - sum(tp))
            for j in used:
                b = size_bin(gts[j]); size_hit[mode][b] = size_hit[mode].get(b, 0) + 1
            if is_neg:
                if mode == "tile_gated" and gate:
                    tiling_fp_on_neg += len(preds)

        # --- gate-atesli alt-kume: no_tile'in kacirdigini tile kurtardi mi ---
        if gate and gts:
            _, nt_used = match(nt_op, gts, args.match_iou)   # nt_op bos -> 0 eslesme
            _, tg_used = match(tg_op, gts, args.match_iou)
            gate_notile_miss += (len(gts) - len(nt_used))
            recovered_fn += (len(tg_used) - len(nt_used))

        # --- AP toplayici (per-image match, low-conf) ---
        for mode, (bs, cs) in (("no_tile", (ff_lb, ff_lc)), ("tile_always", (ta_b, ta_c)),
                               ("tile_only", (tl_lb, tl_lc))):   # tile_only = SAF tile (union YOK) — adil kıyas
            preds = list(zip(bs, cs))
            tp, _ = match(preds, gts, args.match_iou)
            for (b, c), t in zip(preds, tp):
                ap_preds[mode].append((c, t))

        per_frame.append({"stem": s, "src": imgs[s], "n_gt": len(gts), "gate_fired": int(gate)})
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(stems)}  ({(k+1)/(time.time()-t0):.1f} kare/s)")

    # ---- operating-point P/R/F1 ----
    def prf(mode):
        tp, fp = tp_op[mode], fp_op[mode]
        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / total_gt if total_gt else 0.0
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        return P, R, F

    # ---- AP50 ----
    def ap50(mode):
        arr = sorted(ap_preds[mode], key=lambda x: -x[0])
        if not arr or total_gt == 0:
            return 0.0
        tp = np.cumsum([t for _, t in arr]); fp = np.cumsum([1 - t for _, t in arr])
        rec = tp / total_gt; prec = tp / np.maximum(tp + fp, 1e-9)
        return voc_ap(rec.tolist(), prec.tolist())

    P_nt, R_nt, F_nt = prf("no_tile")
    P_tg, R_tg, F_tg = prf("tile_gated")
    AP_nt, AP_ta, AP_to = ap50("no_tile"), ap50("tile_always"), ap50("tile_only")

    # ---- rapor ----
    os.makedirs(args.out, exist_ok=True)
    summary = {
        "n_frames": len(per_frame), "n_negative_frames": n_neg, "total_gt_boxes": total_gt,
        "params": {"conf": args.conf, "second_conf": args.second_conf, "imgsz": args.imgsz,
                   "upscale": args.upscale, "iou": args.iou, "match_iou": args.match_iou,
                   "tile": "2x2 overlap 0.22 (gated: full-frame 0 tespit verince)"},
        "operating_point": {
            "no_tile":    {"P": P_nt, "R": R_nt, "F1": F_nt},
            "tile_gated": {"P": P_tg, "R": R_tg, "F1": F_tg},
            "delta_recall": R_tg - R_nt, "delta_precision": P_tg - P_nt,
        },
        "gate": {"n_gate_fired": n_gate_fired,
                 "on_fired_subset": {"notile_missed_gt": gate_notile_miss, "tile_recovered": recovered_fn}},
        "map50_ceiling": {"no_tile": AP_nt, "tile_always": AP_ta, "tile_only": AP_to, "delta": AP_ta - AP_nt},
        "negatives": {"tiling_extra_fp_on_negatives": tiling_fp_on_neg,
                      "note": ("negatif yok — precision maliyeti OLCULMEDI" if n_neg == 0 else "olculdu")},
        "size_recall": {m: {b: (size_hit[m].get(b, 0) / size_gt[b] if size_gt.get(b) else 0.0)
                             for b in sorted(size_gt)} for m in modes_op},
        "size_gt_counts": size_gt,
    }
    with open(os.path.join(args.out, "AB5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "AB5_per_frame.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["stem", "src", "n_gt", "gate_fired"]); wr.writeheader()
        wr.writerows(per_frame)

    # markdown tablo
    md = []
    md.append("# AB5 — NO-TILE vs TILE (tiling katkisi)\n")
    md.append(f"- Kare: **{len(per_frame)}** ({n_neg} negatif) · GT kutu: **{total_gt}**")
    md.append(f"- Param: conf={args.conf} · second_conf={args.second_conf} · imgsz={args.imgsz} · "
              f"upscale={args.upscale} · tile 2x2 ov0.22\n")
    md.append("## Operating-point (DAGITIM)")
    md.append("| mod | Precision | Recall | F1 |")
    md.append("|---|---|---|---|")
    md.append(f"| no-tile | {P_nt:.3f} | {R_nt:.3f} | {F_nt:.3f} |")
    md.append(f"| **tile (gated)** | {P_tg:.3f} | **{R_tg:.3f}** | {F_tg:.3f} |")
    md.append(f"| **Δ (tiling)** | {P_tg-P_nt:+.3f} | **{R_tg-R_nt:+.3f}** | {F_tg-F_nt:+.3f} |\n")
    md.append("> **Caption:** tiling YALNIZ full-frame conf={:.2f}'de 0 tespit verince atesler "
              "(dagitim davranisi). ΔPrecision = tiling'in negatiflerde FP maliyeti.\n".format(args.conf))
    md.append(f"## Gate analizi (tiling'in gercekten calistigi yer)")
    md.append(f"- Tile yolu ATESLENEN kare: **{n_gate_fired}** / {len(per_frame)}")
    md.append(f"- O alt-kumede: no-tile'in KACIRDIGI GT = {gate_notile_miss} · "
              f"tile'in KURTARDIGI = **{recovered_fn}**\n")
    md.append("## mAP50 tavani (tile-always, dusuk conf)")
    md.append(f"- no-tile AP50 = {AP_nt:.3f} · tile-always AP50 = {AP_ta:.3f} · Δ = **{AP_ta-AP_nt:+.3f}**")
    md.append("> Gated ΔmAP yapisal olarak kucuktur (degismeyen karelerde seyrelir); bu yuzden yukaridaki "
              "gate-alt-kume recovered-FN asil tiling sinyalidir.\n")
    md.append("## Boyut-katmanli recall")
    md.append("| boyut | GT | no-tile R | tile R |")
    md.append("|---|---|---|---|")
    for b in sorted(size_gt):
        rnt = size_hit["no_tile"].get(b, 0) / size_gt[b]
        rtg = size_hit["tile_gated"].get(b, 0) / size_gt[b]
        md.append(f"| {b} | {size_gt[b]} | {rnt:.3f} | {rtg:.3f} |")
    if n_neg == 0:
        md.append("\n> ⚠️ Negatif (bos-label) kare YOK → tiling'in precision maliyeti bu sette OLCULEMEDI.")
    with open(os.path.join(args.out, "AB5_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # figur
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bins = sorted(size_gt)
        x = np.arange(len(bins) + 1)
        rnt = [R_nt] + [size_hit["no_tile"].get(b, 0) / size_gt[b] for b in bins]
        rtg = [R_tg] + [size_hit["tile_gated"].get(b, 0) / size_gt[b] for b in bins]
        labels = ["all"] + bins
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(x - 0.2, rnt, 0.4, label="no-tile", color="#999999")
        ax.bar(x + 0.2, rtg, 0.4, label="2×2 tiled (gated)", color="#0072B2")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Recall @ held-out deployment"); ax.set_ylim(0, 1.12)
        ax.set_title(f"Inference-time tiling: recall by object size (all: {R_tg-R_nt:+.3f})")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
        ax.grid(axis="y", alpha=0.3)
        for xx, (a, b) in enumerate(zip(rnt, rtg)):
            if b - a > 0.005:
                ax.annotate(f"+{b-a:.2f}", (xx, b + 0.02), ha="center", fontsize=8, color="#2f81f7")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "AB5_recall_by_size.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("[uyari] figur uretilemedi:", e)

    print("\n=== AB5 SONUC ===")
    print(f"  operating-point ΔRecall = {R_tg-R_nt:+.3f} (no-tile {R_nt:.3f} -> tile {R_tg:.3f})")
    print(f"  ΔPrecision = {P_tg-P_nt:+.3f} | gate atesli {n_gate_fired} kare, recovered-FN {recovered_fn}")
    print(f"  mAP50: no-tile {AP_nt:.3f} -> tile-always {AP_ta:.3f} (Δ {AP_ta-AP_nt:+.3f})")
    print(f"  -> {args.out}\\AB5_report.md")


if __name__ == "__main__":
    main()

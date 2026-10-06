# -*- coding: utf-8 -*-
r"""E42 ek analiz — senaryo bazinda ESIK SUPURMESI: aksamdaki precision dususu
KALIBRASYON mu, yoksa look-alike gercekten mi zorlasiyor?

AYIRT EDICI TEST: esigi yukseltince fark KAPANIYORSA kalibrasyon (dusuk-guvenli
ateslemeler eleniyor); KAPANMIYORSA yanlis atesler YUKSEK GUVENLI demektir ve
esikle cozulemez.

SONUC (olculdu): SN'de fark kapaniyor (-0.181 -> 0.000 @0.70), SF'te ACILIYOR
(-0.369 -> -0.807). FP sayilari ayni seyi soyluyor: 21:30'da SN yanlislari esikle
eriyor (201->37, %82), SF yanlislari erimiyor (489->323, %34).

⚠️ ATFETME SINIRI: SF'in ayirt edici ozelligi sahnede TUTULAN look-alike olmasi, ama
   look-alike nesne ETIKETLENMEDI (Temmuz protokoluyle ayni; bkz. OKUBENI). Dolayisiyla
   "atesler nesnenin UZERINDE" diyemeyiz — yalniz SENARYO duzeyinde desen raporlanir.

Cikti: runs_v3_eval/angle_experiment/e42_scenario_sweep.csv + fig_e42_scenario_sweep.png
"""
import os
os.environ["PYTHONUTF8"] = "1"
import sys
import csv
import glob
import collections

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import fig_style as S

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
sys.path.insert(0, os.path.join(ROOT, "tools"))
from eval_video_metrics import iou_xyxy   # noqa: E402  (ayni matcher)

UP = os.path.join(ROOT, "CVAT_UPLOAD")
EX = os.path.join(ROOT, "CVAT_EXPORT")
OUT = os.path.join(ROOT, "runs_v3_eval", "angle_experiment")
W, H = 1920, 1080
IOU = 0.30
CONFS = [0.30, 0.40, 0.50, 0.60, 0.70]
SCEN = ["SN", "SF", "MF"]
SCEN_DESC = {"SN": "one smoker + idle bystander", "SF": "one smoker + look-alike",
             "MF": "two smokers + look-alike"}


def y2x(cx, cy, w, h):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def load(tasks, pred_csv, strip=""):
    def norm(k):
        return k[len(strip):] if strip and k.startswith(strip) else k
    gt = {}
    for t in tasks:
        for p in sorted(glob.glob(os.path.join(UP, t, "images", "train", "*.jpg"))):
            gt[norm(os.path.basename(p))] = []
    for t in tasks:
        for f in glob.glob(os.path.join(EX, t, "labels", "**", "*.txt"), recursive=True):
            k = norm(os.path.splitext(os.path.basename(f))[0] + ".jpg")
            if k in gt:
                for ln in open(f, encoding="utf-8"):
                    q = ln.split()
                    if len(q) >= 5:
                        gt[k].append(y2x(*map(float, q[1:5])))
    pr = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(OUT, pred_csv), encoding="utf-8-sig")):
        pr[norm(r["image"])].append(((float(r["x1"]), float(r["y1"]),
                                      float(r["x2"]), float(r["y2"])), float(r["conf"])))
    return gt, pr


def pr_counts(gt, pr, imgs, c):
    TP = FP = FN = 0
    for im in imgs:
        g = list(gt.get(im, []))
        used = set()
        for b, s in sorted([(b, s) for b, s in pr.get(im, []) if s >= c], key=lambda x: -x[1]):
            best, bi = 0.0, -1
            for j, gg in enumerate(g):
                if j in used:
                    continue
                v = iou_xyxy(b, gg)
                if v > best:
                    best, bi = v, j
            if best >= IOU and bi >= 0:
                TP += 1; used.add(bi)
            else:
                FP += 1
        FN += len(g) - len(used)
    P = TP / (TP + FP) if TP + FP else float("nan")
    R = TP / (TP + FN) if TP + FN else float("nan")
    return P, R, TP, FP, FN


gd, pd_ = load([f"aci_A{i}" for i in (1, 2, 3, 4)], "_pred_boxes.csv")
ge, pe = load([f"aksam_A{i}" for i in (1, 2, 3, 4)], "_pred_boxes_aksam.csv", "AK_")

rows = []
print(f"{'sen':4s} {'conf':>5s} | {'P 13:30':>8s} {'P 21:30':>8s} {'dP':>7s} | "
      f"{'FP 13:30':>8s} {'FP 21:30':>8s}")
for s in SCEN:
    imd = [i for i in gd if i.split("_")[1] == s]
    ime = [i for i in ge if i.split("_")[1] == s]
    for c in CONFS:
        p1, r1, t1, f1, n1 = pr_counts(gd, pd_, imd, c)
        p2, r2, t2, f2, n2 = pr_counts(ge, pe, ime, c)
        print(f"{s:4s} {c:5.2f} | {p1:8.3f} {p2:8.3f} {p2-p1:+7.3f} | {f1:8d} {f2:8d}")
        rows.append(dict(senaryo=s, conf=c, prec_1330=round(p1, 4), prec_2130=round(p2, 4),
                         d_prec=round(p2 - p1, 4), rec_1330=round(r1, 4), rec_2130=round(r2, 4),
                         fp_1330=f1, fp_2130=f2, tp_1330=t1, tp_2130=t2))
    print("-" * 72)

with open(os.path.join(OUT, "e42_scenario_sweep.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

# ---- figur: 2 panel, gomulu iddia basligi YOK ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))
col = {"SN": S.OI["green"], "SF": S.OI["verm"], "MF": S.OI["blue"]}
for s in SCEN:
    rs = [r for r in rows if r["senaryo"] == s]
    ax1.plot([r["conf"] for r in rs], [r["d_prec"] for r in rs], "o-", color=col[s],
             lw=1.8, ms=6, label=f"{s} — {SCEN_DESC[s]}")
    base = [r for r in rs if r["conf"] == 0.30][0]["fp_2130"]
    ax1_y = [100.0 * r["fp_2130"] / base for r in rs]
    ax2.plot([r["conf"] for r in rs], ax1_y, "o-", color=col[s], lw=1.8, ms=6)
ax1.axhline(0, color="#444", lw=1)
ax1.set_xlabel("confidence threshold")
ax1.set_ylabel(r"$\Delta$Precision   (21:30 $-$ 13:30)")
ax1.grid(alpha=.3); ax1.set_title("(A) Precision gap against the confidence threshold", fontsize=10, loc="left")
ax2.set_xlabel("confidence threshold")
ax2.set_ylabel("21:30 false positives\n(% of the count at conf 0.30)")
ax2.set_ylim(0, 105); ax2.grid(alpha=.3)
ax2.set_title("(B) How far the false positives thin out", fontsize=10, loc="left")
fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=[0, 0.07, 1, 1])
for d in (os.path.join(ROOT, "runs_v3_eval", "figures"),
          os.path.join(ROOT, "PAPERS", "PAPER1_detector_tiling_temporal", "figures"),
          os.path.join(r"C:\Users\bedrhan94\Desktop\makaleler"
                       r"\Makale1_Detector-Tiling-Temporal", "latex", "figures")):
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "fig_e42_scenario_sweep.png"), dpi=300, bbox_inches="tight")
plt.close()
print("\n[OK] e42_scenario_sweep.csv + fig_e42_scenario_sweep.png")

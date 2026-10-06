# -*- coding: utf-8 -*-
r"""E41 kliplerinin DEDEKTOR tarafi (Paper 1 icin) — dogrulayici YOK.

Bu klipler Paper 2'ye dogrulayici olcumu olarak girdi; ama ayni goruntu dedektor icin de
bir olcumdur ve Paper 1'in kendi eksenine aittir:
  * denek EGITIMDE HIC GORULMEMIS (tasarim beyani)
  * sahne yerlesim deneyinden FARKLI (ic mekan degil, acik teras/balkon)
  * AYNI kisi iki aydinlatmada (13:30 gun isigi / 21:30 yalniz yapay isik)

Protokol Paper 1 ile AYNI: conf 0.30, eslesme IoU 0.30, 1 fps ornekleme, kilitli dedektor,
greedy skor-sirali birebir eslestirme (placement_eval.match ile ayni mantik).

⚠️ FP klibi BURAYA GIRMEZ: GT'si tasarim geregi bos, dedektor precision'ini sonsuz
   cezalandirir ve "sigara iceren sahnede dedektor ne yapiyor" sorusunu bulandirir.
   O klip yalniz dogrulayicinin red kolu icindir (Paper 2).

Cikti: runs_v3_eval/angle_experiment/e41_detector_eval.csv + konsol
"""
import os
os.environ["PYTHONUTF8"] = "1"
import csv
import glob
import sys
import collections

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
sys.path.insert(0, os.path.join(ROOT, "tools"))
from eval_video_metrics import iou_xyxy, compute_ap_single_class   # noqa: E402

UP = os.path.join(ROOT, "CVAT_UPLOAD")
EX = os.path.join(ROOT, "CVAT_EXPORT")
OUT = os.path.join(ROOT, "runs_v3_eval", "angle_experiment")
W_DEF, H_DEF = 1920, 1080
DEPLOY_CONF, IOU_MAIN = 0.30, 0.30
SES = [("13:30", "nobody_sabah_pos", "_pred_boxes_nobody_sabah.csv", "daylight present"),
       ("21:30", "nobody_pos", "_pred_boxes_nobody.csv", "artificial light only")]


def y2x(cx, cy, w, h):
    return ((cx - w/2) * W_DEF, (cy - h/2) * H_DEF, (cx + w/2) * W_DEF, (cy + h/2) * H_DEF)


def load(task, pred_csv):
    gt = {}
    for p in sorted(glob.glob(os.path.join(UP, task, "images", "train", "*.jpg"))):
        gt[os.path.basename(p)] = []            # once hepsi BOS (etiketsiz kare = gercek negatif)
    for f in glob.glob(os.path.join(EX, task, "labels", "**", "*.txt"), recursive=True):
        k = os.path.splitext(os.path.basename(f))[0] + ".jpg"
        if k in gt:
            for ln in open(f, encoding="utf-8"):
                q = ln.split()
                if len(q) >= 5:
                    gt[k].append(y2x(*map(float, q[1:5])))
    pr = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(OUT, pred_csv), encoding="utf-8-sig")):
        im = r["image"]
        if im in gt:                            # FP klibi gt'de YOK -> otomatik dislanir
            pr[im].append(((float(r["x1"]), float(r["y1"]),
                            float(r["x2"]), float(r["y2"])), float(r["conf"])))
    return gt, pr


def match(imgs, gt, pr, conf):
    TP = FP = FN = 0
    for im in imgs:
        g = list(gt.get(im, []))
        used = set()
        for b, s in sorted([(b, s) for b, s in pr.get(im, []) if s >= conf], key=lambda x: -x[1]):
            best, bi = 0.0, -1
            for j, gg in enumerate(g):
                if j in used:
                    continue
                v = iou_xyxy(b, gg)
                if v > best:
                    best, bi = v, j
            if best >= IOU_MAIN and bi >= 0:
                TP += 1; used.add(bi)
            else:
                FP += 1
        FN += len(g) - len(used)
    P = TP / (TP + FP) if TP + FP else float("nan")
    R = TP / (TP + FN) if TP + FN else float("nan")
    F = 2*P*R/(P+R) if P and R and (P+R) > 0 else float("nan")
    return TP, FP, FN, P, R, F


rows = []
print("=" * 94)
print("E41 KLIPLERI — DEDEKTOR (dogrulayici YOK) | conf 0.30, IoU 0.30, kilitli model")
print("hic gorulmemis denek, acik teras; AYNI kisi iki aydinlatmada")
print("=" * 94)
print(f"{'oturum':8}{'kosul':22}{'kare':>6}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
      f"{'P':>8}{'R':>8}{'F1':>8}{'AP@.30':>9}")
pool = {"gt": {}, "pr": {}}
for lbl, task, pc, cond in SES:
    gt, pr = load(task, pc)
    imgs = sorted(gt)
    TP, FP, FN, P, R, F = match(imgs, gt, pr, DEPLOY_CONF)
    gt_by = {i: gt[im] for i, im in enumerate(imgs)}
    preds = [{"frame_idx": i, "box": b, "conf": s}
             for i, im in enumerate(imgs) for b, s in pr.get(im, [])]
    ap = compute_ap_single_class(preds, gt_by, IOU_MAIN)
    print(f"{lbl:8}{cond:22}{len(imgs):6d}{TP+FN:6d}{TP:6d}{FP:6d}{FN:6d}"
          f"{P:8.3f}{R:8.3f}{F:8.3f}{ap:9.3f}")
    rows.append(dict(oturum=lbl, kosul=cond, n_kare=len(imgs), n_gt=TP+FN, TP=TP, FP=FP, FN=FN,
                     precision=round(P, 4), recall=round(R, 4), f1=round(F, 4), ap30=round(ap, 4)))
    pool["gt"].update({f"{lbl}|{k}": v for k, v in gt.items()})
    pool["pr"].update({f"{lbl}|{k}": v for k, v in pr.items()})

imgs = sorted(pool["gt"])
TP, FP, FN, P, R, F = match(imgs, pool["gt"], pool["pr"], DEPLOY_CONF)
gt_by = {i: pool["gt"][im] for i, im in enumerate(imgs)}
preds = [{"frame_idx": i, "box": b, "conf": s}
         for i, im in enumerate(imgs) for b, s in pool["pr"].get(im, [])]
ap = compute_ap_single_class(preds, gt_by, IOU_MAIN)
print("-" * 94)
print(f"{'POOLED':8}{'both sessions':22}{len(imgs):6d}{TP+FN:6d}{TP:6d}{FP:6d}{FN:6d}"
      f"{P:8.3f}{R:8.3f}{F:8.3f}{ap:9.3f}")
rows.append(dict(oturum="POOLED", kosul="both sessions", n_kare=len(imgs), n_gt=TP+FN,
                 TP=TP, FP=FP, FN=FN, precision=round(P, 4), recall=round(R, 4),
                 f1=round(F, 4), ap30=round(ap, 4)))

print("\nESIK SUPURMESI (iki oturum ayri)")
print(f"{'conf':>6}" + "".join(f"{l+' P':>10}{l+' R':>10}" for l, _, _, _ in SES))
sweep = []
data = {lbl: load(t, p) for lbl, t, p, _ in SES}
for c in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
    line = f"{c:6.2f}"
    rec = {"conf": c}
    for lbl, _, _, _ in SES:
        g, pr = data[lbl]
        _, _, _, P, R, _ = match(sorted(g), g, pr, c)
        line += f"{P:10.3f}{R:10.3f}"
        rec[f"{lbl}_P"] = round(P, 4); rec[f"{lbl}_R"] = round(R, 4)
    print(line); sweep.append(rec)

with open(os.path.join(OUT, "e41_detector_eval.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT, "e41_detector_sweep.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)
print("\n[OK] e41_detector_eval.csv / e41_detector_sweep.csv")
print("NOT: FP klibi DISARIDA (GT'si tasarim geregi bos); o klip Paper 2'nin red kolu.")

# -*- coding: utf-8 -*-
r"""DENEY 2 (KONUM) — GT'ye karsi gercek precision/recall.

GT: CVAT_EXPORT/aci_A*/labels/**/*.txt (kullanicinin duzelttigi etiketler)
PR: runs_v3_eval/angle_experiment/_pred_boxes.csv (conf>=0.05, tiled, kilitli model)

KRITIK KURALLAR
1. 420 karenin HEPSI uzerinden donuluyor. Export'ta txt YOKSA = GT bos (gercek negatif),
   "atla" DEGIL. Aksi halde dedektorun FP urettigi kareler (look-alike sahneleri) dusurulur
   ve precision yapay olarak yukselir.
2. Eslesme IoU 0.30 (proje standardi, fig4 ile ayni); IoU 0.50 de raporlanir.
3. Ongorulen esik = dagitim esigi 0.30. Esik supurmesi ayrica verilir.
4. ON-KAYITLI KURAL: bir konum siralamasi UC senaryonun UCUNDE de ayni yonde degilse
   IDDIA EDILMEZ (tekrar varyansi sayilir). Script bunu otomatik uygular.
5. Konum etkisi icin GUVEN ARALIGI HESAPLANMAZ: her (konum x senaryo) hucresinde n=1 cekim
   var; kare/olay uzerinden bootstrap konum farkinin anlamliligini OLCEMEZ, sadece klip-ici
   kesinligi verir. Bunu istatistiksel kanit gibi sunmak yaniltici olur.

Kosma: python tools/paper_analysis/placement_eval.py
"""
import os
os.environ["PYTHONUTF8"] = "1"
import sys
import csv
import glob
import json
import collections

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
sys.path.insert(0, os.path.join(ROOT, "tools"))
from eval_video_metrics import iou_xyxy, compute_ap_single_class   # noqa: E402  (ayni matcher)

UP = os.path.join(ROOT, "CVAT_UPLOAD")
EX = os.path.join(ROOT, "CVAT_EXPORT")
OUT = os.path.join(ROOT, "runs_v3_eval", "angle_experiment")
PRED = os.path.join(OUT, "_pred_boxes.csv")

TASKS = ["aci_A1", "aci_A2", "aci_A3", "aci_A4"]
POS = {"A1": "K1", "A2": "K2", "A3": "K3", "A4": "K4"}
POS_DESC = {"K1": "ana konum (calisilan yer)", "K2": "K1 karsisi, kamera tarafi",
            "K3": "kamera tarafi, sol", "K4": "K3 karsisi"}
SCEN = ["SN", "SF", "MF"]
DEPLOY_CONF = 0.30
IOU_MAIN = 0.30
W_DEF, H_DEF = 1920, 1080


def yolo_to_xyxy(cx, cy, w, h, W, H):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def load_gt():
    """basename -> [xyxy]. TUM 420 kare icin anahtar acilir; txt yoksa bos liste."""
    gt = {}
    for t in TASKS:
        for p in sorted(glob.glob(os.path.join(UP, t, "images", "train", "*.jpg"))):
            gt[os.path.basename(p)] = []          # once hepsi BOS
    n_files = 0
    for t in TASKS:
        for f in glob.glob(os.path.join(EX, t, "labels", "**", "*.txt"), recursive=True):
            key = os.path.splitext(os.path.basename(f))[0] + ".jpg"
            if key not in gt:
                print(f"  [UYARI] export'ta var, kare listesinde yok: {key}")
                continue
            n_files += 1
            for line in open(f, encoding="utf-8"):
                q = line.split()
                if len(q) >= 5:
                    gt[key].append(yolo_to_xyxy(*map(float, q[1:5]), W_DEF, H_DEF))
    n_empty = sum(1 for v in gt.values() if not v)
    print(f"[GT] {len(gt)} kare | etiket dosyasi {n_files} | kutusuz (gercek negatif) {n_empty} "
          f"| toplam kutu {sum(len(v) for v in gt.values())}")
    return gt


def load_pred():
    pr = collections.defaultdict(list)
    with open(PRED, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pr[r["image"]].append(((float(r["x1"]), float(r["y1"]),
                                    float(r["x2"]), float(r["y2"])), float(r["conf"])))
    print(f"[PR] {len(pr)} karede kutu var | toplam {sum(len(v) for v in pr.values())} kutu")
    return pr


def match(images, gt, pred, conf_thr, iou_thr):
    """Greedy skor-sirali birebir eslestirme. Doner: TP, FP, FN."""
    TP = FP = FN = 0
    for im in images:
        g = list(gt.get(im, []))
        pb = sorted([(b, s) for b, s in pred.get(im, []) if s >= conf_thr], key=lambda x: -x[1])
        used = set()
        for b, s in pb:
            best, bi = 0.0, -1
            for j, gg in enumerate(g):
                if j in used:
                    continue
                v = iou_xyxy(b, gg)
                if v > best:
                    best, bi = v, j
            if best >= iou_thr and bi >= 0:
                TP += 1
                used.add(bi)
            else:
                FP += 1
        FN += len(g) - len(used)
    return TP, FP, FN


def prf(TP, FP, FN):
    P = TP / (TP + FP) if TP + FP else float("nan")
    R = TP / (TP + FN) if TP + FN else float("nan")
    F = 2 * P * R / (P + R) if P and R and (P + R) > 0 else float("nan")
    return P, R, F


def main():
    # AKSAM tekrari (E42): ayni kod yolundan gecsin diye parametrelendirildi.
    # VARSAYILANLAR Temmuz davranisini AYNEN korur.
    global TASKS, PRED
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--pred", default=os.path.basename(PRED))
    ap.add_argument("--tag", default="", help="cikti dosya adi soneki (ornek: _aksam)")
    ap.add_argument("--strip-prefix", default="", dest="strip_prefix",
                    help="kare adindan soyulacak onek (aksam icin 'AK_') -> subset() ayni kalir")
    args = ap.parse_args()
    TASKS = [t.strip() for t in args.tasks.split(",") if t.strip()]
    PRED = os.path.join(OUT, args.pred)
    TAG = args.tag

    def _norm(d):
        """Anahtarlardan onegi soy: 'AK_A1_SN_f..' -> 'A1_SN_f..' (subset/3-3 kurali degismesin)."""
        if not args.strip_prefix:
            return d
        n = type(d)()
        for k, v in d.items():
            n[k[len(args.strip_prefix):] if k.startswith(args.strip_prefix) else k] = v
        return n

    gt = _norm(load_gt())
    pred = _norm(load_pred())
    all_imgs = sorted(gt)

    def subset(ang=None, scen=None):
        return [im for im in all_imgs
                if (ang is None or im.startswith(ang + "_"))
                and (scen is None or im.split("_")[1] == scen)]

    rows = []
    print(f"\n{'='*78}\nKONUM x SENARYO — conf={DEPLOY_CONF} (dagitim), IoU={IOU_MAIN}\n{'='*78}")
    print(f"{'konum':6s} {'sen':4s} {'kare':>5s} {'GT':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} "
          f"{'P':>7s} {'R':>7s} {'F1':>7s}")
    cell = {}
    for a in ["A1", "A2", "A3", "A4"]:
        for s in SCEN:
            ims = subset(a, s)
            TP, FP, FN = match(ims, gt, pred, DEPLOY_CONF, IOU_MAIN)
            P, R, F = prf(TP, FP, FN)
            cell[(POS[a], s)] = (P, R, F)
            print(f"{POS[a]:6s} {s:4s} {len(ims):5d} {TP+FN:5d} {TP:5d} {FP:5d} {FN:5d} "
                  f"{P:7.3f} {R:7.3f} {F:7.3f}")
            rows.append(dict(konum=POS[a], senaryo=s, n_kare=len(ims), n_gt=TP + FN,
                             TP=TP, FP=FP, FN=FN, precision=round(P, 4),
                             recall=round(R, 4), f1=round(F, 4),
                             conf=DEPLOY_CONF, iou=IOU_MAIN))

    # --- konum toplami ---
    print(f"\n{'-'*78}\nKONUM TOPLAMI (3 senaryo birlikte)")
    print(f"{'konum':6s} {'GT':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'P':>7s} {'R':>7s} "
          f"{'F1':>7s} {'AP@.30':>7s} {'AP@.50':>7s}")
    tot = {}
    for a in ["A1", "A2", "A3", "A4"]:
        ims = subset(a)
        TP, FP, FN = match(ims, gt, pred, DEPLOY_CONF, IOU_MAIN)
        P, R, F = prf(TP, FP, FN)
        gt_by = {i: gt[im] for i, im in enumerate(ims)}
        preds = [{"frame_idx": i, "box": b, "conf": s}
                 for i, im in enumerate(ims) for b, s in pred.get(im, [])]
        ap30 = compute_ap_single_class(preds, gt_by, 0.30)
        ap50 = compute_ap_single_class(preds, gt_by, 0.50)
        tot[POS[a]] = (P, R, F, ap30, ap50)
        print(f"{POS[a]:6s} {TP+FN:5d} {TP:5d} {FP:5d} {FN:5d} {P:7.3f} {R:7.3f} {F:7.3f} "
              f"{ap30:7.3f} {ap50:7.3f}")
        rows.append(dict(konum=POS[a], senaryo="TOPLAM", n_kare=len(ims), n_gt=TP + FN,
                         TP=TP, FP=FP, FN=FN, precision=round(P, 4), recall=round(R, 4),
                         f1=round(F, 4), ap30=round(ap30, 4), ap50=round(ap50, 4),
                         conf=DEPLOY_CONF, iou=IOU_MAIN))

    # --- ON-KAYITLI 3/3 KURALI ---
    print(f"\n{'='*78}\nON-KAYITLI KURAL: siralama UC senaryoda da ayni mi?\n{'='*78}")
    verdict = {}
    for mi, mname in [(1, "recall"), (0, "precision")]:
        order = {s: sorted(["K1", "K2", "K3", "K4"], key=lambda k: -cell[(k, s)][mi]) for s in SCEN}
        best = {order[s][0] for s in SCEN}
        worst = {order[s][-1] for s in SCEN}
        full = len({tuple(order[s]) for s in SCEN}) == 1
        print(f"\n{mname.upper()} siralamasi:")
        for s in SCEN:
            print(f"  {s}: " + " > ".join(f"{k}({cell[(k,s)][mi]:.3f})" for k in order[s]))
        okb = len(best) == 1
        okw = len(worst) == 1
        print(f"  -> EN IYI 3/3 tutarli mi? {'EVET: '+list(best)[0] if okb else 'HAYIR ('+','.join(sorted(best))+') -> IDDIA EDILMEZ'}")
        print(f"  -> EN KOTU 3/3 tutarli mi? {'EVET: '+list(worst)[0] if okw else 'HAYIR ('+','.join(sorted(worst))+') -> IDDIA EDILMEZ'}")
        print(f"  -> TAM siralama ayni mi? {'EVET' if full else 'HAYIR'}")
        verdict[mname] = dict(best=sorted(best), worst=sorted(worst),
                              best_consistent=okb, worst_consistent=okw, full_order_same=full,
                              per_scenario={s: order[s] for s in SCEN})

    # --- esik supurmesi (konum toplami, recall) ---
    print(f"\n{'='*78}\nESIK SUPURMESI (konum toplami, IoU {IOU_MAIN})\n{'='*78}")
    sweep = []
    print(f"{'conf':>6s} " + " ".join(f"{POS[a]+'_P':>8s} {POS[a]+'_R':>8s}" for a in ['A1','A2','A3','A4']))
    for c in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
        line = f"{c:6.2f} "
        rec = {"conf": c}
        for a in ["A1", "A2", "A3", "A4"]:
            TP, FP, FN = match(subset(a), gt, pred, c, IOU_MAIN)
            P, R, _ = prf(TP, FP, FN)
            line += f" {P:8.3f} {R:8.3f}"
            rec[POS[a] + "_P"] = round(P, 4)
            rec[POS[a] + "_R"] = round(R, 4)
        sweep.append(rec)
        print(line)

    # --- senaryo kirilimi (DENEY 3 baglantisi): SF = look-alike tuzagi ---
    print(f"\n{'='*78}\nSENARYO TOPLAMI (konumlar birlikte) — SF = look-alike tuzagi\n{'='*78}")
    print(f"{'sen':5s} {'GT':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    for s in SCEN:
        ims = subset(None, s)
        TP, FP, FN = match(ims, gt, pred, DEPLOY_CONF, IOU_MAIN)
        P, R, F = prf(TP, FP, FN)
        print(f"{s:5s} {TP+FN:5d} {TP:5d} {FP:5d} {FN:5d} {P:7.3f} {R:7.3f} {F:7.3f}")
        rows.append(dict(konum="TOPLAM", senaryo=s, n_kare=len(ims), n_gt=TP + FN, TP=TP,
                         FP=FP, FN=FN, precision=round(P, 4), recall=round(R, 4),
                         f1=round(F, 4), conf=DEPLOY_CONF, iou=IOU_MAIN))

    # --- yaz ---
    cols = ["konum", "senaryo", "n_kare", "n_gt", "TP", "FP", "FN",
            "precision", "recall", "f1", "ap30", "ap50", "conf", "iou"]
    with open(os.path.join(OUT, "placement_stratified" + TAG + ".csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "placement_sweep" + TAG + ".csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)
    with open(os.path.join(OUT, "placement_verdict" + TAG + ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(deploy_conf=DEPLOY_CONF, iou=IOU_MAIN,
                       pos_desc=POS_DESC, cells={f"{k[0]}_{k[1]}": v for k, v in cell.items()},
                       totals=tot, prereg_rule=verdict), f, indent=2, ensure_ascii=False)
    print("\n[OK] placement_stratified.csv / placement_sweep.csv / placement_verdict.json")
    print("\nNOT: Konum farki icin guven araligi verilmedi - her (konum x senaryo) hucresinde")
    print("     n=1 cekim var; konum etkisi tekrar varyansiyla ic ice. Tek dayanak 3/3 kurali.")


if __name__ == "__main__":
    main()

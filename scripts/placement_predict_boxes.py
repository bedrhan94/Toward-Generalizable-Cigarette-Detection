# -*- coding: utf-8 -*-
r"""DENEY 2 (KONUM) — etiketlenmis 1680 kare uzerinde KUTU-SEVIYESI tahmin uretir.

Neden ayri script: `angle_detect_batch.py` sadece kare-basi ozet (n_det/max_conf) yaziyordu;
precision/recall icin kutu koordinati + skor lazim. Ayrica esik SUPURULEBILSIN diye conf=0.05
ile kosuyoruz (dagitim esigi 0.30 sonradan filtreyle uygulanir).

Kareler VIDEODAN YENIDEN CIKARILMAZ — kullanicinin etiketledigi TAM AYNI jpg'ler okunur
(CVAT_UPLOAD/aci_A*/images/train/*.jpg). Boylece GT ile tahmin ayni piksellerden gelir.

Tiling/model/imgsz `angle_cvat_prefill.py` ile BIREBIR ayni (kilitli model SHA 17BF8...E3D14).

Cikti: runs_v3_eval/angle_experiment/_pred_boxes.csv
       image,angle,scenario,W,H,x1,y1,x2,y2,conf     (kutusuz kare icin satir yazilmaz)
"""
import os
os.environ["PYTHONUTF8"] = "1"
import glob
import csv
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
UP = ROOT + r"\CVAT_UPLOAD"
OUT = ROOT + r"\runs_v3_eval\angle_experiment"
MODEL = ROOT + r"\models\yolo11m_full_dataset_finetune_e38_best.pt"
IMGSZ = 960
OVERLAP = 0.2
MERGE_IOU = 0.5


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def nms(boxes, scores, thr=MERGE_IOU):
    if len(boxes) == 0:
        return []
    idx = list(np.argsort(scores)[::-1])
    keep = []
    while idx:
        i = idx[0]
        keep.append(i)
        idx = [j for j in idx[1:] if iou(boxes[i], boxes[j]) < thr]
    return keep


def detect_tiled(model, fr, conf):
    """2x2 ortusmeli tile + tam kare, sonra tile-NMS. prefill scriptiyle birebir ayni."""
    H, W = fr.shape[:2]
    th, tw = int(H * (0.5 + OVERLAP / 2)), int(W * (0.5 + OVERLAP / 2))
    allb, alls = [], []
    passes = [(0, 0, W, H)] + [(ox, oy, ox + tw, oy + th)
                               for oy in (0, H - th) for ox in (0, W - tw)]
    for (x1, y1, x2, y2) in passes:
        r = model.predict(fr[y1:y2, x1:x2], imgsz=IMGSZ, conf=conf, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                allb.append([b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1])
                alls.append(float(s))
    allb = np.array(allb) if allb else np.zeros((0, 4))
    alls = np.array(alls) if alls else np.zeros(0)
    k = nms(allb, alls)
    return (allb[k] if len(k) else np.zeros((0, 4))), (alls[k] if len(k) else np.zeros(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.05,
                    help="TABAN esik (dusuk tutulur; dagitim esigi sonradan filtreyle uygulanir)")
    # AKSAM tekrari (E42) icin parametrelendirildi; VARSAYILANLAR Temmuz davranisi.
    ap.add_argument("--glob", default="aci_A*", help="CVAT_UPLOAD altinda task deseni")
    ap.add_argument("--out", default="_pred_boxes.csv", help="OUT altinda cikti dosyasi")
    ap.add_argument("--name-offset", type=int, default=0, dest="name_offset",
                    help="dosya adinda konum kodunun indeksi (Temmuz 'A1_SN_f..'=0, aksam 'AK_A1_SN_f..'=1)")
    args = ap.parse_args()

    model = YOLO(MODEL)
    os.makedirs(OUT, exist_ok=True)
    rows = [("image", "angle", "scenario", "W", "H", "x1", "y1", "x2", "y2", "conf")]
    n_img = 0

    for t in sorted(glob.glob(os.path.join(UP, args.glob))):
        if not os.path.isdir(t):
            continue
        imgs = sorted(glob.glob(os.path.join(t, "images", "train", "*.jpg")))
        for p in tqdm(imgs, desc=os.path.basename(t), ncols=88):
            fr = imread_u(p)
            if fr is None:
                print(f"  ATLANDI (okunamadi): {p}", flush=True)
                continue
            H, W = fr.shape[:2]
            b, s = detect_tiled(model, fr, args.conf)
            name = os.path.basename(p)
            _pp = name.split("_")
            ang, scen = _pp[args.name_offset], _pp[args.name_offset + 1]
            for (x1, y1, x2, y2), sc in zip(b, s):
                rows.append((name, ang, scen, W, H,
                             f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", f"{sc:.4f}"))
            n_img += 1

    dst = os.path.join(OUT, args.out)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n[OK] {dst}  |  {n_img} kare, {len(rows)-1} kutu (conf>={args.conf})", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""AB5 (EĞİTİM-zamanı) — no_tile'dan AYNI source'tan FULL ve TILE eğitim seti kur.

TASARIM (danışman: training-time izolasyon, eval koşulu iki kolda AYNI):
  * Çekim (t##) bazında böl -> TRAIN çekimleri + HELD-OUT çekimleri (kare bazında DEĞİL: bitişik
    kareler sızar). İki kol da AYNI split'i paylaşır -> tiling DELTA'sı adil.
  * FULL  = train çekimlerinin tam kareleri (+label).
  * TILE  = AYNI train karelerinin 2x2 tile'ları (overlap 0.22, deployment'la birebir) + label remap.
  * HELD-OUT = ayrı çekimler, TAM KARE (iki model de burada infer-FULL ile ölçülür -> eğitim-zamanı izole).
  * data.yaml'lar: full.yaml / tile.yaml, val = AYNI heldout (tam kare).

KOSMA:
  .\.venv\Scripts\python.exe tools\ab5_build_traintime.py --src no_tile --out ab5_traintime
"""
from __future__ import annotations
import argparse, os, re, shutil, random
from collections import defaultdict
import cv2, numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def tile_windows(w, h, gx=2, gy=2, ov=0.22):
    tw, th = int(np.ceil(w / gx)), int(np.ceil(h / gy))
    sx, sy = int(max(1, tw * (1 - ov))), int(max(1, th * (1 - ov)))
    xs = list(range(0, max(1, w - tw + 1), sx)) or [0]
    ys = list(range(0, max(1, h - th + 1), sy)) or [0]
    out = []
    for y1 in ys:
        for x1 in xs:
            x2, y2 = min(w, x1 + tw), min(h, y1 + th)
            out.append((max(0, x2 - tw), max(0, y2 - th), x2, y2))
    return out


def read_gt(lp, w, h):
    boxes = []
    if not lp or not os.path.exists(lp):
        return boxes
    for ln in open(lp, encoding="utf-8", errors="replace"):
        p = ln.split()
        if len(p) < 5:
            continue
        try:
            cx, cy, bw, bh = float(p[1]) * w, float(p[2]) * h, float(p[3]) * w, float(p[4]) * h
        except ValueError:
            continue
        boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def _wr_img(path, img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        buf.tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="no_tile")
    ap.add_argument("--out", default="ab5_traintime")
    ap.add_argument("--heldout-frac", type=float, default=0.30)
    ap.add_argument("--min-vis", type=float, default=0.30, help="tile'da GT görünürlük eşiği")
    ap.add_argument("--overlap", type=float, default=0.22)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    imgdir, lbldir = os.path.join(a.src, "images"), os.path.join(a.src, "labels")
    imgs = {os.path.splitext(f)[0]: os.path.join(imgdir, f)
            for f in os.listdir(imgdir) if f.lower().endswith(IMG_EXT)}
    lbls = {os.path.splitext(f)[0]: os.path.join(lbldir, f)
            for f in os.listdir(lbldir) if f.endswith(".txt")}

    # çekim bazında grupla
    rec = defaultdict(list)
    for s in imgs:
        m = re.match(r"(t\d+)_", s)
        rec[m.group(1) if m else "OTHER"].append(s)
    recs = sorted(rec)
    random.Random(a.seed).shuffle(recs)
    n_hold = max(1, round(len(recs) * a.heldout_frac))
    heldout_recs = set(sorted(recs[:n_hold]))
    train_recs = set(sorted(recs[n_hold:]))
    print(f"[split] {len(recs)} çekim -> train {len(train_recs)} | heldout {len(heldout_recs)}")
    print(f"  heldout: {sorted(heldout_recs)}")

    # klasörler
    def mk(*p):
        d = os.path.join(a.out, *p); os.makedirs(d, exist_ok=True); return d
    full_img, full_lab = mk("full", "images"), mk("full", "labels")
    tile_img, tile_lab = mk("tile", "images"), mk("tile", "labels")
    ho_img, ho_lab = mk("heldout", "images"), mk("heldout", "labels")

    n_full = n_tile = n_ho = n_tilebox = 0
    for s, ip in imgs.items():
        m = re.match(r"(t\d+)_", s); r = m.group(1) if m else "OTHER"
        lp = lbls.get(s)
        if r in heldout_recs:
            # HELD-OUT: tam kare kopyala
            shutil.copy2(ip, os.path.join(ho_img, os.path.basename(ip)))
            if lp: shutil.copy2(lp, os.path.join(ho_lab, s + ".txt"))
            else: open(os.path.join(ho_lab, s + ".txt"), "w").close()
            n_ho += 1
            continue
        # TRAIN: (1) FULL tam kare
        shutil.copy2(ip, os.path.join(full_img, os.path.basename(ip)))
        if lp: shutil.copy2(lp, os.path.join(full_lab, s + ".txt"))
        else: open(os.path.join(full_lab, s + ".txt"), "w").close()
        n_full += 1
        # (2) TILE: aynı kareyi 2x2 böl + label remap
        img = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = read_gt(lp, W, H)
        for ti, (x1, y1, x2, y2) in enumerate(tile_windows(W, H, 2, 2, a.overlap)):
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tw, th = x2 - x1, y2 - y1
            lines = []
            for (bx1, by1, bx2, by2) in gts:
                ix1, iy1 = max(bx1, x1), max(by1, y1)
                ix2, iy2 = min(bx2, x2), min(by2, y2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                barea = (bx2 - bx1) * (by2 - by1)
                if barea <= 0 or (iw * ih) / barea < a.min_vis:
                    continue
                cx = ((ix1 + ix2) / 2 - x1) / tw; cy = ((iy1 + iy2) / 2 - y1) / th
                nw = iw / tw; nh = ih / th
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_tilebox += 1
            base = f"{s}__t{ti}"
            _wr_img(os.path.join(tile_img, base + ".jpg"), crop)
            open(os.path.join(tile_lab, base + ".txt"), "w").write("\n".join(lines))
            n_tile += 1
        if n_full % 500 == 0:
            print(f"  ... {n_full} train kare işlendi")

    # data.yaml'lar (val = AYNI heldout, tam kare)
    ho_abs = os.path.abspath(os.path.join(a.out, "heldout", "images"))
    for arm in ("full", "tile"):
        y = os.path.join(a.out, f"{arm}.yaml")
        tr = os.path.abspath(os.path.join(a.out, arm, "images"))
        open(y, "w", encoding="utf-8").write(
            f"# AB5 training-time ablation — {arm} kolu\n"
            f"train: {tr}\nval: {ho_abs}\nnc: 1\nnames:\n  0: cigarette\n")
        print(f"[yaml] {y}")

    print(f"\n=== KURULDU ===")
    print(f"  FULL train kare : {n_full}")
    print(f"  TILE train tile : {n_tile}  ({n_tilebox} kutu)")
    print(f"  HELD-OUT kare   : {n_ho}  (val, iki kol da burada infer-FULL)")
    print(f"  -> {a.out}/  (full.yaml, tile.yaml)")


if __name__ == "__main__":
    main()

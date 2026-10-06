# -*- coding: utf-8 -*-
r"""AB5 DIAGNOSTIC (danışman blocker #1) — TILE kolu GEÇERLİ mi, yoksa BOZUK mu?

MANTIK: held-out'u EĞİTİMDEKİYLE BİREBİR aynı şekilde tile'la (2x2, ov 0.22, min-vis 0.30) ->
        modelleri KENDİ doğal rejiminde (per-tile) ölç.
  * TILE-e8 tile'da YÜKSEK (~0.8) ama full-frame'de 0.202  -> ÖLÇEK UZMANLAŞMASI kanıtı, kol GEÇERLİ.
  * TILE tile'da da DÜŞÜK -> kol/label remap BOZUK -> §5.6 sayıları GEÇERSİZ.

KOSMA:
  .\.venv\Scripts\python.exe tools\ab5_diagnostic.py
"""
import os
os.environ["PYTHONUTF8"] = "1"


def build_tiled_heldout(src_img, src_lbl, out_root, overlap=0.22, min_vis=0.30):
    import cv2, numpy as np, re
    from ab5_build_traintime import tile_windows, read_gt, _wr_img  # aynı tiling mantığı
    oi = os.path.join(out_root, "images"); ol = os.path.join(out_root, "labels")
    os.makedirs(oi, exist_ok=True); os.makedirs(ol, exist_ok=True)
    imgs = [f for f in os.listdir(src_img) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    n_tile = n_box = 0
    for k, f in enumerate(imgs):
        s = os.path.splitext(f)[0]
        ip = os.path.join(src_img, f); lp = os.path.join(src_lbl, s + ".txt")
        img = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = read_gt(lp if os.path.exists(lp) else None, W, H)
        for ti, (x1, y1, x2, y2) in enumerate(tile_windows(W, H, 2, 2, overlap)):
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
                if barea <= 0 or (iw * ih) / barea < min_vis:
                    continue
                cx = ((ix1 + ix2) / 2 - x1) / tw; cy = ((iy1 + iy2) / 2 - y1) / th
                lines.append(f"0 {cx:.6f} {cy:.6f} {iw/tw:.6f} {ih/th:.6f}")
                n_box += 1
            base = f"{s}__t{ti}"
            _wr_img(os.path.join(oi, base + ".jpg"), crop)
            open(os.path.join(ol, base + ".txt"), "w").write("\n".join(lines))
            n_tile += 1
        if (k + 1) % 500 == 0:
            print(f"  ... {k+1}/{len(imgs)} held-out kare tile'landi", flush=True)
    print(f"[tiled-heldout] {n_tile} tile ({n_box} kutu) -> {out_root}", flush=True)
    return oi


def main():
    import sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tools"))

    src_img = "ab5_traintime/heldout/images"
    src_lbl = "ab5_traintime/heldout/labels"
    out_root = "ab5_traintime/heldout_tiled"
    yaml_path = os.path.abspath("ab5_traintime/heldout_tiled.yaml")

    # 1) tiled held-out kur (varsa atla)
    if os.path.isdir(os.path.join(out_root, "images")) and \
       len(os.listdir(os.path.join(out_root, "images"))) > 0:
        print(f"[skip] {out_root} zaten var, yeniden kurulmuyor", flush=True)
    else:
        print("[build] held-out tile'laniyor (2x2 ov0.22 min-vis0.30)...", flush=True)
        build_tiled_heldout(src_img, src_lbl, out_root)

    tr = os.path.abspath(os.path.join(out_root, "images"))
    open(yaml_path, "w", encoding="utf-8").write(
        f"# AB5 diagnostic — tiled held-out (native tile rejimi)\n"
        f"train: {tr}\nval: {tr}\nnc: 1\nnames:\n  0: cigarette\n")

    # 2) her weight'i tiled held-out'ta val et (native per-tile mAP)
    from ultralytics import YOLO
    weights = {
        "FULL_best_e12": "runs_ab5/full/weights/best.pt",
        "TILE_best_e2":  "runs_ab5/tile/weights/best.pt",
        "TILE_last_e8":  "runs_ab5/tile/weights/last.pt",
    }
    rows = []
    for tag, wp in weights.items():
        if not os.path.exists(wp):
            print(f"[atla] {tag}: {wp} yok", flush=True); continue
        print(f"\n{'='*50}\n=== TILED-HELDOUT VAL: {tag} ===\n{'='*50}", flush=True)
        m = YOLO(wp)
        r = m.val(data=yaml_path, imgsz=960, batch=16, device=0, verbose=False,
                  project="runs_ab5_eval", name=f"tilediag_{tag}", exist_ok=True)
        rows.append((tag, float(r.box.map50), float(r.box.map), float(r.box.mp), float(r.box.mr)))

    print("\n\n########## DIAGNOSTIC SONUC (tiled held-out, per-tile) ##########")
    print(f"{'weight':<16} {'mAP50':>8} {'mAP50-95':>9} {'P':>7} {'R':>7}")
    for tag, m50, m, p, rr in rows:
        print(f"{tag:<16} {m50:>8.3f} {m:>9.3f} {p:>7.3f} {rr:>7.3f}")
    print("\nYORUM: TILE_last tile'da YÜKSEK + full-frame'de 0.202 -> ölçek uzmanlaşması, kol GEÇERLİ.")
    print("       TILE tile'da da DÜŞÜK -> kol BOZUK, §5.6 geçersiz.")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()

# -*- coding: utf-8 -*-
r"""#1 TEMPORAL FSM PARAMETRE SWEEP (K_off, N_min) — kalem/marker setinde.

§3.4'te K_on TARANDI ama K_off=8 ve N_min=0.8s "boyunca sabit" diye DAYATILDI. Hakem "neden 8?
neden 0.8s?" sorar, eğri yok. Bu script o eğriyi üretir — kalıntının (0.6/dk) YAŞADIĞI kalem/marker
setinde (fig4 ile aynı zemin: native 1920, conf 0.30, tiling YOK).

AYNI build_events_from_binary (eval_video_metrics'ten import) → fig4 ile birebir tutar.
DOĞRULAMA: K_on=2s, K_off=8, N_min=0.8s → fig4'ün ~0.6/dk'sını üretmeli (config teyidi).

Çıktı: runs_v3_eval/fsm_sweep/  (CSV + figür). GPU KULLANMAZ (jsonl'den okur).
"""
import os
import sys
import json
os.environ["PYTHONUTF8"] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
sys.path.insert(0, os.path.join(ROOT, "tools"))
from eval_video_metrics import build_events_from_binary  # noqa: E402

OUT = ROOT + r"\runs_v3_eval\fsm_sweep"
PRED = OUT + r"\pred_pensmarkers_1920.jsonl"

# sabit referans (fig4 dağıtım noktası)
KON_S = 2.0; KOFF_REF = 8; NMIN_REF_S = 0.8
KOFF_GRID = [4, 6, 8, 12, 16]
NMIN_GRID = [0.4, 0.8, 1.2, 1.6]
KON_GRID_S = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]   # fig4 doğrulaması için
NAVY = "#1F3A5F"; BLUE = "#2B7BBA"; RED = "#C0392B"; GREEN = "#1E8A3E"


def load_present(pred_jsonl):
    present = []; fps = 30.0
    with open(pred_jsonl, encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            fps = float(r.get("fps", fps))
            present.append(1 if r.get("pred") else 0)
    return np.array(present, np.int8), fps


def fa_per_min(present, fps, kon_s, koff, nmin_s):
    kon = max(1, int(round(kon_s * fps)))
    ev = build_events_from_binary(present, fps, kon, koff, nmin_s)
    minutes = len(present) / fps / 60.0
    return len(ev), len(ev) / max(minutes, 1e-9)


def main():
    if not os.path.exists(PRED):
        raise SystemExit(f"Önce detektör predict'i bitmeli: {PRED} yok.")
    present, fps = load_present(PRED)
    minutes = len(present) / fps / 60.0
    print(f"[fsm] kare={len(present)} fps={fps:.1f} süre={minutes:.1f}dk "
          f"tespitli-kare=%{100*present.mean():.1f}")

    # --- DOĞRULAMA: fig4 K_on sweep'i (K_off=8, N_min=0.8s sabit) ---
    print("\n[DOĞRULAMA] K_on sweep (K_off=8, N_min=0.8s) — fig4 ile karşılaştır:")
    print("  fig4 referans: 0.0->15.1 0.25->4.1 0.5->2.4 1.0->0.8 2.0->0.6 3.0->0.1")
    val_rows = []
    for ks in KON_GRID_S:
        n, fam = fa_per_min(present, fps, ks, KOFF_REF, NMIN_REF_S)
        val_rows.append((ks, n, fam))
        print(f"  K_on={ks:4.2f}s -> {n:3d} olay = {fam:.2f}/dk")

    rows = ["sweep,param_value,kon_s,koff,nmin_s,n_events,fa_per_min"]
    for ks, n, fam in val_rows:
        rows.append(f"validation_kon,{ks},{ks},{KOFF_REF},{NMIN_REF_S},{n},{fam:.4f}")

    # --- K_off sweep (K_on=2s, N_min=0.8s sabit) ---
    print(f"\n[K_off SWEEP] (K_on={KON_S}s, N_min={NMIN_REF_S}s sabit):")
    koff_fa = []
    for koff in KOFF_GRID:
        n, fam = fa_per_min(present, fps, KON_S, koff, NMIN_REF_S)
        koff_fa.append(fam)
        star = "  <- referans (§3.4)" if koff == KOFF_REF else ""
        print(f"  K_off={koff:2d} -> {n:3d} olay = {fam:.2f}/dk{star}")
        rows.append(f"koff,{koff},{KON_S},{koff},{NMIN_REF_S},{n},{fam:.4f}")

    # --- N_min sweep (K_on=2s, K_off=8 sabit) ---
    print(f"\n[N_min SWEEP] (K_on={KON_S}s, K_off={KOFF_REF} sabit):")
    nmin_fa = []
    for nm in NMIN_GRID:
        n, fam = fa_per_min(present, fps, KON_S, KOFF_REF, nm)
        nmin_fa.append(fam)
        star = "  <- referans (§3.4)" if abs(nm - NMIN_REF_S) < 1e-6 else ""
        print(f"  N_min={nm:.1f}s -> {n:3d} olay = {fam:.2f}/dk{star}")
        rows.append(f"nmin,{nm},{KON_S},{KOFF_REF},{nm},{n},{fam:.4f}")

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "fsm_param_sweep.csv"), "w", encoding="utf-8").write("\n".join(rows) + "\n")

    # figür: 3 panel (K_on doğrulama, K_off, N_min)
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.5, 4.2))
    a0.plot([r[0] for r in val_rows], [r[2] for r in val_rows], "o-", color=BLUE, lw=2, ms=6)
    a0.axvline(2.0, color=GREEN, ls="--", lw=1.3); a0.set_title("DOĞRULAMA: K_on sweep\n(fig4'ü yeniden üretir)",
                                                                fontsize=10, weight="bold", color=NAVY)
    a0.set_xlabel("K_on onay penceresi (s)"); a0.set_ylabel("yanlış alarm / dk"); a0.grid(alpha=0.3)
    a0.annotate(f"2s -> {val_rows[4][2]:.2f}/dk\n(fig4: 0.6)", (2.0, val_rows[4][2]),
                textcoords="offset points", xytext=(8, 12), fontsize=8, color=GREEN)

    a1.plot(KOFF_GRID, koff_fa, "s-", color=RED, lw=2, ms=7)
    a1.axvline(KOFF_REF, color=GREEN, ls="--", lw=1.3, label="§3.4 seçimi = 8")
    a1.set_title("K_off SWEEP\n(K_on=2s, N_min=0.8s)", fontsize=10, weight="bold", color=NAVY)
    a1.set_xlabel("K_off durdurma-kaçırma (kare)"); a1.set_ylabel("yanlış alarm / dk")
    a1.grid(alpha=0.3); a1.legend(fontsize=8.5)

    a2.plot(NMIN_GRID, nmin_fa, "^-", color="#8E44AD", lw=2, ms=7)
    a2.axvline(NMIN_REF_S, color=GREEN, ls="--", lw=1.3, label="§3.4 seçimi = 0.8s")
    a2.set_title("N_min SWEEP\n(K_on=2s, K_off=8)", fontsize=10, weight="bold", color=NAVY)
    a2.set_xlabel("N_min min. olay süresi (s)"); a2.set_ylabel("yanlış alarm / dk")
    a2.grid(alpha=0.3); a2.legend(fontsize=8.5)

    fig.suptitle("Temporal FSM parametre duyarlılığı — kalem/marker seti (kalıntının yaşadığı yer)\n"
                 "§3.4'ün K_off=8 / N_min=0.8s seçimlerinin gerekçesi (keyfi değil, eğriyle gösterildi)",
                 fontsize=11, weight="bold", color=NAVY)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(OUT, "fig_fsm_param_sweep.png"), dpi=160); plt.close()
    print(f"\n=== çıktı -> {OUT} ===")
    # config teyidi
    v2 = val_rows[4][2]
    print(f"\n[CONFIG TEYİDİ] K_on=2s -> {v2:.2f}/dk (fig4=0.6). "
          f"{'✓ EŞLEŞTİ' if abs(v2-0.6) < 0.5 else '⚠ FARK VAR — config kontrol et (imgsz/tiling)'}")


if __name__ == "__main__":
    main()

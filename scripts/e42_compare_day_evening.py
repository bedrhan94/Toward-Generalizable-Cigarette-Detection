# -*- coding: utf-8 -*-
r"""E42 — 13:30 (gunduz) vs 21:30 (aksam) KARSILASTIRMASI, hucre hucre.

Iki cekim de AYNI kod yolundan gecti (placement_eval.py; Temmuz sayilari birebir yeniden
uretildi = kapi gecildi), bu yuzden farklar koddan DEGIL veriden.

⚠️ n=2 CI VERMEZ. Her (konum x senaryo) hucresinde iki cekim var; bu, cekimler-arasi
   FARKI raporlamaya yeter, guven araligina yetmez. E34'un "verilemez" kavati tamamen
   kalkmiyor. Ayrica GT kutu sayisi hucrelerde degisti (sigara daha az/cok karede gorundu)
   -> farkin bir kismi dedektor degil TEKRAR VARYANSI. GT farki her satirda yaninda.

⚠️ ZAMAN EKSENI ADI: 13:30 ogleden SONRA (sabah DEGIL), 21:30'da hava karanlik ve oda iki
   cepheden camli => "gun isigi mevcut" vs "yalniz yapay isik". Makalede saat yazilacak.

Cikti: runs_v3_eval/angle_experiment/e42_day_vs_evening.csv + konsol tablosu
"""
import os
os.environ["PYTHONUTF8"] = "1"
import csv
import io
import sys
import collections

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
OUT = os.path.join(ROOT, "runs_v3_eval", "angle_experiment")
DAY = os.path.join(OUT, "placement_stratified.csv")
EVE = os.path.join(OUT, "placement_stratified_aksam.csv")
SCEN = ["SN", "SF", "MF"]
POS = ["K1", "K2", "K3", "K4"]


def load(p):
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        d[(r["konum"], r["senaryo"])] = r
    return d


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


day, eve = load(DAY), load(EVE)
rows = []

print("=" * 108)
print("E42 — HUCRE BAZINDA 13:30 (gun isigi) vs 21:30 (yalniz yapay isik)")
print("=" * 108)
print(f"{'hucre':9s} {'GT 13:30':>9s} {'GT 21:30':>9s} {'dGT':>5s} | "
      f"{'R 13:30':>8s} {'R 21:30':>8s} {'dR':>7s} | {'P 13:30':>8s} {'P 21:30':>8s} {'dP':>7s}")
for k in POS:
    for s in SCEN:
        a, b = day.get((k, s)), eve.get((k, s))
        if not (a and b):
            continue
        g1, g2 = f(a, "n_gt"), f(b, "n_gt")
        r1, r2 = f(a, "recall"), f(b, "recall")
        p1, p2 = f(a, "precision"), f(b, "precision")
        print(f"{k}/{s:4s} {g1:9.0f} {g2:9.0f} {g2-g1:+5.0f} | "
              f"{r1:8.3f} {r2:8.3f} {r2-r1:+7.3f} | {p1:8.3f} {p2:8.3f} {p2-p1:+7.3f}")
        rows.append(dict(konum=k, senaryo=s, gt_1330=int(g1), gt_2130=int(g2), d_gt=int(g2-g1),
                         recall_1330=round(r1, 4), recall_2130=round(r2, 4), d_recall=round(r2-r1, 4),
                         prec_1330=round(p1, 4), prec_2130=round(p2, 4), d_prec=round(p2-p1, 4)))
    a, b = day.get((k, "TOPLAM")), eve.get((k, "TOPLAM"))
    if a and b:
        r1, r2 = f(a, "recall"), f(b, "recall")
        p1, p2 = f(a, "precision"), f(b, "precision")
        a1, a2 = f(a, "ap30"), f(b, "ap30")
        print(f"{k}/TOPLAM {f(a,'n_gt'):9.0f} {f(b,'n_gt'):9.0f} {f(b,'n_gt')-f(a,'n_gt'):+5.0f} | "
              f"{r1:8.3f} {r2:8.3f} {r2-r1:+7.3f} | {p1:8.3f} {p2:8.3f} {p2-p1:+7.3f}   "
              f"AP@.30 {a1:.3f} -> {a2:.3f} ({a2-a1:+.3f})")
        rows.append(dict(konum=k, senaryo="TOPLAM", gt_1330=int(f(a, "n_gt")), gt_2130=int(f(b, "n_gt")),
                         d_gt=int(f(b, "n_gt") - f(a, "n_gt")),
                         recall_1330=round(r1, 4), recall_2130=round(r2, 4), d_recall=round(r2-r1, 4),
                         prec_1330=round(p1, 4), prec_2130=round(p2, 4), d_prec=round(p2-p1, 4),
                         ap30_1330=round(a1, 4), ap30_2130=round(a2, 4), d_ap30=round(a2-a1, 4)))
    print("-" * 108)

# --- ON-KAYITLI SORU: K4-en-yuksek-recall AKSAMDA da 3/3 mu? ---
print("\n" + "=" * 108)
print("ON-KAYITLI SORU — 'bir konum siralamasi UC senaryonun UCUNDE de ayni yonde mi?'")
print("=" * 108)
for lbl, d in (("13:30 (gun isigi)", day), ("21:30 (yapay isik)", eve)):
    for met in ("recall", "precision"):
        best = []
        for s in SCEN:
            vals = [(k, f(d[(k, s)], met)) for k in POS if (k, s) in d]
            best.append(max(vals, key=lambda x: x[1])[0] if vals else None)
        ok = len(set(best)) == 1
        print(f"  {lbl:20s} {met:10s} en iyi: {' / '.join(f'{s}={b}' for s, b in zip(SCEN, best))}"
              f"   -> 3/3 {'GECTI: ' + best[0] if ok else 'GECMEDI'}")

print("\nSONUC (ozet):")
print("  * 13:30'da K4 recall'da 3/3 GECMISTI (E34'un tek raporlanabilir yonsel bulgusu).")
print("  * 21:30'da bu TEKRARLANMIYOR: SN ve SF'te K3, yalniz MF'te K4 onde.")
print("  * Dolayisiyla yerlesim-recall siralamasi gun-ici-saate/aydinlatmaya DAYANIKLI DEGIL.")
print("  * Zaten raporlanamayan siralamalar (precision, en-kotu konum) iki cekimde de 3/3 gecmiyor.")
print("\nDIKKAT: Bu bir NEGATIF bulgu ve oyle yazilacak. Ayrica aksamda dedektor ayni esikte")
print("        (conf 0.30) COK daha fazla atesliyor: recall yukseliyor, precision cokuyor")
print("        -> yerlesim degil KALIBRASYON kaymasi; Paper 1 §5.6 (ECE 0.312) ile tutarli.")

with open(os.path.join(OUT, "e42_day_vs_evening.csv"), "w", newline="", encoding="utf-8") as fh:
    keys = ["konum", "senaryo", "gt_1330", "gt_2130", "d_gt", "recall_1330", "recall_2130",
            "d_recall", "prec_1330", "prec_2130", "d_prec", "ap30_1330", "ap30_2130", "d_ap30"]
    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(f"\n[OK] {os.path.join(OUT, 'e42_day_vs_evening.csv')}")

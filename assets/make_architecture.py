"""Render assets/architecture.png.

Drawn with Pillow at 2x and downsampled, so the PNG stays crisp on HiDPI
screens. Dark card background with light text — readable on both the GitHub
light and dark themes.

Run:  python assets/make_architecture.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2  # supersample factor
W, H = 940 * S, 560 * S
OUT = Path(__file__).with_name("architecture.png")

BG = (13, 17, 23)
FG = (201, 209, 217)
MUTED = (139, 148, 158)
LINE = (110, 118, 129)
ACCENT = (188, 140, 255)
GREEN = (63, 185, 80)

FONTS = r"C:\Windows\Fonts"


def font(name, size):
    return ImageFont.truetype(f"{FONTS}\\{name}", size * S)


f_title = font("seguisb.ttf", 15)
f_head = font("seguisb.ttf", 13)
f_small = font("segoeui.ttf", 11)
f_lbl = font("segoeuii.ttf", 10)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def box(x, y, w, h, colour=LINE, width=2):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S],
                        radius=6 * S, outline=colour, width=int(width * S))


def text(x, y, s, f=f_small, fill=MUTED, anchor="mm"):
    d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)


def arrow(pts, colour=LINE, width=1.6, dash=False):
    pts = [(x * S, y * S) for x, y in pts]
    for i in range(len(pts) - 1):
        if dash:
            _dashed(pts[i], pts[i + 1], colour, width)
        else:
            d.line([pts[i], pts[i + 1]], fill=colour, width=int(width * S))
    _head(pts[-2], pts[-1], colour)


def _dashed(p0, p1, colour, width, on=6, off=4):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    while pos < dist:
        seg = min(on * S, dist - pos)
        d.line([(x0 + ux * pos, y0 + uy * pos),
                (x0 + ux * (pos + seg), y0 + uy * (pos + seg))],
               fill=colour, width=int(width * S))
        pos += (on + off) * S


def _head(p0, p1, colour, size=7):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    s = size * S
    d.polygon([(x1, y1),
               (x1 - ux * s + px * s * 0.5, y1 - uy * s + py * s * 0.5),
               (x1 - ux * s - px * s * 0.5, y1 - uy * s - py * s * 0.5)],
              fill=colour)


text(470, 26, "Fraud-Detection-MLOps — training, registry, and the drift → retrain loop",
     f_title, FG)

# ── data sources ────────────────────────────────────────────────────────────
box(24, 52, 170, 66)
text(109, 74, "PaySim", f_head, FG)
text(109, 92, "6.36M rows · simulated")
text(109, 108, "training")

box(212, 52, 170, 66)
text(297, 74, "Sparkov", f_head, FG)
text(297, 92, "1.30M · 2019-01→2020-06")
text(297, 108, "training window")

box(400, 52, 188, 66, GREEN)
text(494, 74, "sparkov_test", f_head, FG)
text(494, 92, "555,719 · 2020-06→2020-12")
text(494, 108, "held-out out-of-time")

# ── features ────────────────────────────────────────────────────────────────
arrow([(109, 118), (109, 150)])
arrow([(297, 118), (297, 150)])
box(24, 152, 358, 72)
text(203, 176, "Feature engineering — Dask / pandas", f_head, FG)
text(203, 195, "bit-exact across engines (max abs diff 5.5e-12)")
text(203, 212, "pandas 1.15M rows/s · Dask 0.26M rows/s on one host")

# ── temporal split ──────────────────────────────────────────────────────────
arrow([(382, 188), (410, 188)])
box(412, 152, 176, 72, ACCENT)
text(500, 176, "Temporal split", f_head, FG)
text(500, 195, "per source, chronological")
text(500, 212, "final 20% held out")

# ── train ───────────────────────────────────────────────────────────────────
arrow([(500, 224), (500, 248)])
box(330, 250, 340, 72)
text(500, 274, "Train — XGBoost + Optuna (30 trials)", f_head, FG)
text(500, 293, "source-balanced: paysim 0.60× · sparkov 2.95×")
text(500, 310, "OOT AUC 0.9520")

# ── registry ────────────────────────────────────────────────────────────────
arrow([(500, 322), (500, 350)])
box(330, 352, 340, 72, ACCENT)
text(500, 376, "MLflow Model Registry", f_head, FG)
text(500, 395, "promote / rollback via alias · audited")
text(500, 412, "alias flip 3.9 ms · audited rollback 11.9 ms")

# ── serving ─────────────────────────────────────────────────────────────────
arrow([(670, 388), (714, 388)])
box(718, 352, 198, 72)
text(817, 376, "FastAPI serving :8000", f_head, FG)
text(817, 395, "alias “production”")
text(817, 412, "60 µs/query · $0.43 per 1k qps/day")

# ── drift monitor ───────────────────────────────────────────────────────────
arrow([(817, 424), (817, 450)])
box(600, 454, 316, 72, ACCENT)
text(758, 477, "Drift monitor — KS + PSI", f_head, FG)
text(758, 496, "8 features · precision 1.0 · recall 1.0")
text(758, 512, "0-day detection lag on a 2σ shift")

# ── retrain loop ────────────────────────────────────────────────────────────
arrow([(600, 490), (484, 490)])
box(164, 454, 316, 72)
text(322, 477, "Auto-retrain → shadow eval → promote", f_head, FG)
text(322, 496, "promote only if shadow AUPRC ≥ prod − 0.010")
text(322, 512, "median 6.85 s end-to-end · 3/3 promoted")

arrow([(164, 490), (96, 490), (96, 388), (328, 388)], ACCENT, dash=True)
text(150, 440, "registers new version", f_lbl, MUTED, anchor="lm")

# held-out set is only ever scored against
arrow([(588, 85), (700, 85), (700, 286), (674, 286)], GREEN, dash=True)
text(710, 178, "scored against only,", f_lbl, GREEN, anchor="lm")
text(710, 194, "never trained on", f_lbl, GREEN, anchor="lm")

text(24, 544, "DVC orchestrates every stage · MLflow records every run",
     f_lbl, MUTED, anchor="lm")

img.resize((W // S, H // S), Image.LANCZOS).save(OUT, "PNG", optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")

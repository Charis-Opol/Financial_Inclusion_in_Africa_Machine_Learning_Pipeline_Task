"""Resize selected report figures and save them as WebP for the website.

Run from the repository root with an environment that has Pillow
(e.g. the XAI venv):

    .venv-xai/Scripts/python website/scripts/export_figures.py .

Writes website/public/figures/*.webp. Captions and alt text live in
website/data/content.ts; update them if a figure changes.
"""
import sys
from pathlib import Path

from PIL import Image

repo = Path(sys.argv[1])
out = repo / "website" / "public" / "figures"
out.mkdir(parents=True, exist_ok=True)

FIGURES = {
    "pr-curve-overlay": "reports/figures/pr_curve_overlay.png",
    "confusion-matrix-xgb-cw": "reports/figures/confusion_matrix_xgb_cw.png",
    "shap-beeswarm-xgboost-pooled": "reports/figures/xai/shap_beeswarm__xgboost__pooled.png",
    "importance-all-methods-pooled": "reports/figures/xai/internals_global_importance_all_methods__pooled.png",
    "shap-waterfall-xgboost-uganda": "reports/figures/xai/shap_waterfall__xgboost__focus_typical.png",
}
MAX_W = 1600

for name, rel in FIGURES.items():
    im = Image.open(repo / rel).convert("RGB")
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    dest = out / f"{name}.webp"
    im.save(dest, "WEBP", quality=88, method=6)
    print(f"{name}: {im.width}x{im.height} {dest.stat().st_size // 1024} KB")

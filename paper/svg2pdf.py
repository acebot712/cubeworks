#!/usr/bin/env python
"""Convert the exported figure SVGs to vector PDF for LaTeX."""
import pathlib
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).parent/'figures'; OUT.mkdir(exist_ok=True)
for n in (2, 8, 9, 10, 12, 13, 14, 15):
    src = ROOT/'eval'/'figures'/f'figure-{n}.svg'
    if src.exists():
        renderPDF.drawToFile(svg2rlg(str(src)), str(OUT/f'fig{n}.pdf'))
        print(f"  figures/fig{n}.pdf")

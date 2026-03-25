"""
print_markers.py  (fixed)
-------------------------
Generates a correct printable SVG of ArUco markers IDs 0-3.

Usage:
    python print_markers.py --diameter 22 --out markers.svg
    python print_markers.py --diameter 22 --size 40 --out markers.svg

Arguments:
    --diameter   Handle/base diameter in mm
    --size       Marker size in mm (overrides diameter-based sizing)
    --margin     White border around each marker in mm (default 3)
    --out        Output SVG filename
"""

import argparse, math, sys
try:
    import cv2, cv2.aruco as aruco, numpy as np
except ImportError:
    print("pip install opencv-python numpy"); sys.exit(1)

def get_bits(mid):
    """Returns 2D list: 0=black, 1=white for a 6x6 DICT_4X4_50 marker."""
    d   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    img = aruco.generateImageMarker(d, mid, 6)   # 6x6 px image
    return (img > 128).astype(int).tolist()       # 0=black, 1=white

def make_svg(ids, marker_mm, margin_mm, out_path):
    scale   = 96 / 25.4          # mm → px at 96 dpi
    cell_mm = marker_mm / 6      # 6x6 grid
    slot_mm = marker_mm + margin_mm * 2
    gap_mm  = 4
    cols    = 2
    rows    = math.ceil(len(ids) / cols)
    page_w  = cols * slot_mm + (cols + 1) * gap_mm
    page_h  = rows * slot_mm + (rows + 1) * gap_mm + 14  # 14 for title

    W = page_w * scale
    H = page_h * scale

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W:.1f}px" height="{H:.1f}px" viewBox="0 0 {W:.1f} {H:.1f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{W/2:.1f}" y="{10*scale:.1f}" text-anchor="middle" '
        f'font-family="monospace" font-size="{3.5*scale:.1f}" fill="#333">'
        f'DICT_4X4_50 — {marker_mm:.1f}mm markers — print at 100%</text>',
    ]

    for i, mid in enumerate(ids):
        col = i % cols
        row = i // cols

        # slot top-left in mm
        sx = gap_mm + col * (slot_mm + gap_mm)
        sy = 14 + gap_mm + row * (slot_mm + gap_mm)

        # dashed cut guide
        lines.append(
            f'<rect x="{sx*scale:.2f}" y="{sy*scale:.2f}" '
            f'width="{slot_mm*scale:.2f}" height="{slot_mm*scale:.2f}" '
            f'fill="white" stroke="#bbb" stroke-width="0.6" stroke-dasharray="3,2"/>'
        )

        # marker top-left (inside margin)
        mx = (sx + margin_mm) * scale
        my = (sy + margin_mm) * scale
        mw = marker_mm * scale
        cs = cell_mm * scale

        # white base
        lines.append(
            f'<rect x="{mx:.2f}" y="{my:.2f}" '
            f'width="{mw:.2f}" height="{mw:.2f}" fill="white"/>'
        )

        # draw black cells
        bits = get_bits(mid)
        for r, rowbits in enumerate(bits):
            for c, bit in enumerate(rowbits):
                if bit == 0:   # black
                    lines.append(
                        f'<rect x="{mx + c*cs:.2f}" y="{my + r*cs:.2f}" '
                        f'width="{cs:.2f}" height="{cs:.2f}" fill="black"/>'
                    )

        # label
        lx = (sx + slot_mm/2) * scale
        ly = (sy + slot_mm + 3.5) * scale
        lines.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'font-family="monospace" font-size="{3*scale:.1f}" fill="#444">'
            f'ID {mid}</text>'
        )

    lines.append('</svg>')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved: {out_path}")
    print(f"Marker size : {marker_mm:.1f} x {marker_mm:.1f} mm")
    print(f"Cell size   : {cell_mm:.1f} mm")
    print("Open in browser → print at 100% (disable 'fit to page')")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--diameter", type=float, default=None, help="Handle diameter mm (sets marker size to circumference/4)")
    p.add_argument("--size",     type=float, default=None, help="Marker size in mm (overrides --diameter)")
    p.add_argument("--ids",      nargs="+",  type=int, default=[0,1,2,3])
    p.add_argument("--margin",   type=float, default=3.0)
    p.add_argument("--out",      default="markers.svg")
    a = p.parse_args()

    if a.size:
        marker_mm = a.size
    elif a.diameter:
        marker_mm = (math.pi * a.diameter) / 4
    else:
        marker_mm = 20.0   # sensible default

    make_svg(a.ids, marker_mm, a.margin, a.out)
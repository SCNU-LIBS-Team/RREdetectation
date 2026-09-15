# -*- coding: utf-8 -*-
"""English flowchart of the temperature-scanning confidence test.

Same conventions as Paper/Plot/figure5/generate_temperature_iteration_flowchart.py:
data units are PostScript points (1 unit = 1 pt) so font metrics can be
reasoned about directly; all node text is 12 pt Arial bold with 1.15 line
spacing.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
})

LW = 2.0            # node / arrow line width (pt)
FS = 12             # node text size (pt), bold
LSP = 1.15          # node text line spacing

W, H = 990, 170     # canvas size in pt
fig = plt.figure(figsize=(W / 72, H / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

_nodes = {}         # name -> (text artist, cx, cy, w, h, kind)


def box(name, cx, cy, w, h, text, rounded=False):
    style = "round,pad=0,rounding_size=10" if rounded else "square,pad=0"
    p = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h, boxstyle=style,
        fc="white", ec="black", lw=LW, mutation_aspect=1)
    ax.add_patch(p)
    t = ax.text(cx, cy, text, ha="center", va="center", fontsize=FS,
                fontweight="bold", linespacing=LSP)
    _nodes[name] = (t, cx, cy, w, h, "box")


def diamond(name, cx, cy, w, h, text):
    p = mpatches.Polygon(
        [(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)],
        closed=True, fc="white", ec="black", lw=LW)
    ax.add_patch(p)
    t = ax.text(cx, cy, text, ha="center", va="center", fontsize=FS,
                fontweight="bold", linespacing=LSP)
    _nodes[name] = (t, cx, cy, w, h, "diamond")


def label(text, x, y, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=FS, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1))


def _arrow_patch(p1, p2):
    ax.add_patch(mpatches.FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=10,
        color="black", lw=LW, shrinkA=0, shrinkB=0))


def arrow(points):
    """Polyline arrow with rounded corners; arrowhead on the last segment."""
    pts = [np.array(p, float) for p in points]
    verts = [tuple(pts[0])]
    codes = [Path.MOVETO]
    R = 8
    for i in range(1, len(pts) - 1):
        p, v1, v2 = pts[i], pts[i - 1] - pts[i], pts[i + 1] - pts[i]
        rr = min(R, np.hypot(*v1) / 2, np.hypot(*v2) / 2)
        a = p + v1 / np.hypot(*v1) * rr
        b = p + v2 / np.hypot(*v2) * rr
        verts += [tuple(a), tuple(p), tuple(b)]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(tuple(pts[-1]))
    codes.append(Path.LINETO)
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), fc="none",
                                    ec="black", lw=LW, joinstyle="round"))
    tail = tuple(pts[-2]) if len(pts) == 2 else verts[-2]
    _arrow_patch(tail, tuple(pts[-1]))


# ================= nodes: main row (y grows upward) =================
Y = 130

box("input", 80, Y, 140, 44, "Input elements\nwith confidence", rounded=True)
box("retain", 250, Y, 140, 44, "Retain initial\nspectral lines")
box("vary", 425, Y, 150, 56, "Vary temperature\n5000-20000 K\nin 250 K steps")
box("fluct", 595, Y, 130, 44, "Confidence\nfluctuation Δ")
diamond("test", 745, Y, 110, 70, "Δ > 0.5?")
box("output", 910, Y, 140, 44, "Output element\nconfidence", rounded=True)

arrow([(150, Y), (180, Y)])
arrow([(320, Y), (350, Y)])
arrow([(500, Y), (530, Y)])
arrow([(660, Y), (690, Y)])
arrow([(800, Y), (840, Y)])
label("Yes", 820, Y + 15)

# No branch: not confirmed -> mark as false detection, zero the confidence.
box("false", 745, 45, 160, 44, "False detection\nConfidence set to 0")
arrow([(745, Y - 35), (745, 67)])
label("No", 756, 81, ha="left")


# ================= text-fit self check =================
def check_fits():
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad = []
    for name, (t, cx, cy, w, h, kind) in _nodes.items():
        bb = t.get_window_extent(renderer)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        tw, th = x1 - x0, y1 - y0
        if kind == "box":
            ok = tw <= w - 6 and th <= h - 6
        else:  # diamond: text block must clear the slanted sides
            ok = tw <= w * (1 - th / h) and th <= h * 0.75
        if not ok:
            bad.append(f"{name}: text {tw:.0f}x{th:.0f}pt vs {kind} {w}x{h}pt")
    if bad:
        print("TEXT OVERFLOW:")
        for b in bad:
            print("  " + b)
    else:
        print("all node texts fit")


check_fits()

fig.savefig("温度扫描_横版.svg")
fig.savefig("温度扫描_横版.pdf")
fig.savefig("温度扫描_横版.png", dpi=600)
fig.savefig("温度扫描_横版.jpg", dpi=600, pil_kwargs={"quality": 95})
print("saved svg / pdf / png / jpg")

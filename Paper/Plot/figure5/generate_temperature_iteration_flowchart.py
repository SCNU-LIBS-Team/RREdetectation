# -*- coding: utf-8 -*-
"""English flowchart of the temperature-iteration identification algorithm.

Faithful translation of the original Chinese flowchart (温度迭代流程.jpg):
two disconnected panels, each wrapped in a coloured dashed frame.

Data units are PostScript points (1 unit = 1 pt) so font metrics can be
reasoned about directly; all node text is 12 pt Arial bold.
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
FLW = 2.5           # dashed frame line width (pt)
FS = 12             # node text size (pt), bold
LSP = 1.15          # node text line spacing

W, H = 800, 556     # canvas size in pt
fig = plt.figure(figsize=(W / 72, H / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

_nodes = {}         # name -> (text artist, cx, cy, w, h, kind)


def frame(x0, y0, x1, y1, color, lw=FLW):
    p = mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=10", fc="none", ec=color,
        lw=lw, ls=(0, (7, 5)), mutation_aspect=1)
    ax.add_patch(p)


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


def arrow_v(a, b, gap_only=False):
    """Vertical arrow from bottom edge of node a to top edge of node b."""
    _, cx, cy, w, h, _ = _nodes[a]
    arrow([(cx, cy - h / 2), (cx, cy - h / 2 - gap_only)])


def arrow_v_to(a, b):
    _, acx, acy, aw, ah, _ = _nodes[a]
    _, bcx, bcy, bw, bh, _ = _nodes[b]
    arrow([(acx, acy - ah / 2), (bcx, bcy + bh / 2)])


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


# ================= dashed panel frames =================
frame(6, 6, 452, 550, "#D73027")      # left: outer loop panel
frame(472, 67, 793.5, 489, "#B2D990")  # right: inner loop panel

# ================= left panel: outer temperature loop =================
LX = 245.5

box("L1", LX, 517, 150, 50, "Outer temperature\niteration loop", rounded=True)
box("L2", LX, 431, 205, 70, "Generate multiple starting\npoints (10-point uniform\nsampling over 5000–20000 K)")
box("L3", LX, 345, 184, 50, "Invoke inner temperature\niteration")
diamond("L4", LX, 249, 205, 90, "Is the current\nscore the best\nso far?")
diamond("L5", LX, 133, 200, 90, "Any unvisited\ntemperature starting\npoints?")
box("L6", LX, 37, 130, 50, "Output global\noptimum $T$", rounded=True)
box("L4Y", 72, 249, 121, 50, "Update the best\ntemperature")
box("L4N", 394, 249, 72, 50, "Continue")

arrow_v_to("L1", "L2")
arrow_v_to("L2", "L3")
arrow_v_to("L3", "L4")
arrow_v_to("L4", "L5")
arrow_v_to("L5", "L6")
label("No", LX, 74)
arrow([(143, 249), (132.5, 249)])      # Yes -> update the best temperature
label("Yes", 137.5, 262)
arrow([(348, 249), (358, 249)])        # No -> continue
label("No", 353, 262)
# Yes: unvisited starting points remain -> back into the invoke step
arrow([(340.5, 133), (442, 133), (442, 345), (337.5, 345)])
label("Yes", 442, 290)

# green dashed frame: this step invokes the inner loop (right panel)
frame(141.5, 313, 349.5, 377, "#B2D990")

# ================= right panel: inner temperature loop =================
RX = 630.5

box("R1", RX, 463, 150, 40, "Inner temperature\niteration loop", rounded=True)
box("R2", RX, 415, 205, 40, "Retrieve the matrix-element\nspectral line library")
box("R3", RX, 367, 174, 40, "Wavelet peak detection\nwith confidence output")
box("R4", RX, 308, 150, 62, "Compute the target\ntemperature via\nSoftmax-weighted\nTOP-3 candidates")
box("R5", RX, 249, 128, 40, "Damped update of\nthe temperature")
diamond("R6", RX, 166, 292, 110, "Has the temperature\nconverged? ($\\Delta$ < tolerance\nfor two consecutive\nrounds?)")
box("R7", RX, 93, 170, 40, "Output the final\ninner-loop temperature", rounded=True)

arrow_v_to("R1", "R2")
arrow_v_to("R2", "R3")
arrow_v_to("R3", "R4")
arrow_v_to("R4", "R5")
arrow_v_to("R5", "R6")
arrow_v_to("R6", "R7")
label("Yes", RX, 117)
# No: not converged -> back into the wavelet peak detection step
arrow([(776.5, 166), (788.5, 166), (788.5, 367), (717.5, 367)])
label("No", 776, 269)


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

fig.savefig("temperature_iteration_flowchart.svg")
fig.savefig("temperature_iteration_flowchart.pdf")
fig.savefig("temperature_iteration_flowchart.png", dpi=600)
print("saved svg / pdf / png")

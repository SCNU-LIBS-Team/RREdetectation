# -*- coding: utf-8 -*-
"""Objective comparison between the reproduced figure and the reference.

Usage:
    python compare.py [repro.png] [ref.png]

Defaults:
    repro = output/example1_repro.png (next to this script)
    ref   = ../example1.png (the original survey image)

Checks (thresholds from SPEC.md section 6):
  1. identical canvas size (1253 x 800);
  2. pure-blue pixel count within +-12% of the reference survey value 6562;
  3. orange pixels in the repro (rows < 740) must be 0 -- the orange Tm II
     curve of the original is intentionally NOT drawn (user request);
  4. blue curve centerline deviation: RMSE <= 2.5 px and p95 <= 6 px
     (steep columns with pure-color runs > 10 px and the label region
     rows >= 740 are excluded), reference-column coverage >= 95%, and
     repro-only columns <= 5% of reference columns;
  5. red curve centerline deviation, same rule, computed only on columns
     where the reference contains red pixels (the reference red baseline is
     partially hidden under the orange curve), reference-column coverage >=
     95%, and repro-only columns <= 58% of reference columns;
  6. major/minor grid line centers detected with the same gray-pixel
     centroid detector on both images, deviation <= 2 px;
  7. x tick label cluster centers (7 clusters, the first one clipped)
     deviation <= 3 px;
  8. background spot checks are white.

The script prints one PASS/FAIL line per check and finishes with
``RESULT: PASS`` or ``RESULT: FAIL``.  Exit code is 0 on PASS, 1 on FAIL.
"""

from __future__ import absolute_import, print_function

import os
import sys

import numpy as np
from PIL import Image

_script_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_REPRO = os.path.join(_script_dir, "output", "example1_repro.png")
DEFAULT_REF = os.path.join(_script_dir, os.pardir, "example1.png")

CANVAS = (800, 1253)  # (rows, cols)

BLUE = (0x33, 0x66, 0xCC)
RED = (0xDC, 0x39, 0x12)
ORANGE = (0xFF, 0x99, 0x00)

BLUE_PURE_TARGET = 6562   # survey value; blue is unoccluded in the reference
PURE_TOLERANCE = 0.12

AXIS_BOTTOM_ROW = 727.5   # y = 0 baseline sits here
LABEL_REGION_ROW = 740    # rows >= 740 belong to the x tick label zone
STEEP_RUN_PX = 10         # column runs longer than this are "steep"
CENTER_RMSE_MAX = 2.5
CENTER_P95_MAX = 6.0
CURVE_REFERENCE_COVERAGE_MIN = 0.95
BLUE_REPRO_ONLY_RATIO_MAX = 0.05
# The reference orange curve hides much of the red baseline, while the
# requested reproduction intentionally omits orange and therefore exposes it.
RED_REPRO_ONLY_RATIO_MAX = 0.58
CURVE_COMMON_COLUMNS_MIN = 100

GRID_CENTER_MAX = 2.0     # grid line center deviation
CLUSTER_CENTER_MAX = 3.0  # tick label cluster center deviation
NUM_CLUSTERS = 7

WHITE_FRAC_MIN = 0.85
WHITE_FRAC_DIFF = 0.05
SPOT_POINTS = [(3, 3), (3, 1249), (400, 1249), (735, 100), (740, 300),
               (790, 626), (790, 1100)]


def load_image(path):
    """Load an image as an (H, W, 3) int array."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.int64)


def exact_mask(image, color):
    """Boolean mask of pixels exactly matching ``color``."""
    return np.all(image == np.array(color, dtype=np.int64), axis=2)


def curve_centers(image, color):
    """Per-column centerline of a pure-color curve.

    Returns a dict {col: center_row}.  A column is skipped when its matching
    rows form a contiguous run longer than STEEP_RUN_PX (steep segments) or
    when it lies in the label region (rows >= LABEL_REGION_ROW are ignored).
    """
    mask = exact_mask(image, color)
    mask[LABEL_REGION_ROW:, :] = False
    centers = {}
    for col in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, col])
        if rows.size == 0:
            continue
        splits = np.flatnonzero(np.diff(rows) > 1)
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits, [rows.size - 1]))
        lengths = ends - starts + 1
        if lengths.max() > STEEP_RUN_PX:
            continue
        centers[col] = float(rows.mean())
    return centers


def center_deviation(centers_a, centers_b, restrict=None):
    """Signed deviations a-b over common columns (optionally restricted)."""
    cols = set(centers_a) & set(centers_b)
    if restrict is not None:
        cols &= set(restrict)
    cols = sorted(cols)
    devs = np.array([centers_a[c] - centers_b[c] for c in cols], dtype=float)
    return cols, devs


def gray_profile(image, axis, row_limit):
    """Darkness profile of grayish (low-saturation, mid-bright) pixels.

    ``axis='v'`` sums over rows 0..row_limit and returns a per-column
    profile; ``axis='h'`` sums over columns and returns a per-row profile.
    Only gray pixels count, so the colored curves and the dark labels are
    excluded automatically.
    """
    region = image[:row_limit, :]
    spread = region.max(axis=2) - region.min(axis=2)
    value = region.mean(axis=2)
    grayish = (spread <= 12) & (value >= 140) & (value <= 246)
    darkness = np.where(grayish, 255.0 - value, 0.0)
    return darkness.sum(axis=0 if axis == "v" else 1)


def profile_groups(profile, threshold):
    """Contiguous runs of profile > threshold as (lo, hi) index pairs."""
    idx = np.flatnonzero(profile > threshold)
    if idx.size == 0:
        return []
    groups = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > 2:  # gap -> new line
            groups.append((start, prev))
            start = i
        prev = i
    groups.append((start, prev))
    return groups


def detect_lines(profile, extent, density_split=35.0):
    """Detect grid lines in a darkness profile.

    Returns (majors, minors): each a list of darkness-weighted sub-pixel
    centers.  A candidate group's density is its peak darkness divided by
    the profile extent; major #CCCCCC lines score ~2.5x the minor #EBEBEB
    ones in both images.
    """
    hits = []
    for lo, hi in profile_groups(profile, 3000.0):
        weights = profile[lo:hi + 1]
        positions = np.arange(lo, hi + 1, dtype=float)
        center = float((positions * weights).sum() / weights.sum())
        density = float(weights.max()) / float(extent)
        hits.append((center, density))
    majors = [c for c, d in hits if d >= density_split]
    minors = [c for c, d in hits if d < density_split]
    return majors, minors


def detect_grid(image):
    """Grid line centers of an image -> dict with 4 sorted center lists."""
    v_profile = gray_profile(image, "v", int(AXIS_BOTTOM_ROW) + 1)
    h_profile = gray_profile(image, "h", 690)
    v_major, v_minor = detect_lines(v_profile, int(AXIS_BOTTOM_ROW) + 1)
    h_major, h_minor = detect_lines(h_profile, image.shape[1])
    return {"major_v": sorted(v_major), "minor_v": sorted(v_minor),
            "major_h": sorted(h_major), "minor_h": sorted(h_minor)}


def max_match_dev(list_r, list_f):
    """Max deviation under ordered one-to-one center matching.

    Grid centers lie on one dimension, so sorted pairwise matching is the
    minimum-cost one-to-one assignment.  A nearest-neighbour loop is unsafe:
    it can reuse one detected line for several reference lines.
    """
    if len(list_r) != len(list_f):
        return None
    if not list_r:
        return 0.0
    return max(abs(c_r - c_f)
               for c_r, c_f in zip(sorted(list_r), sorted(list_f)))


def curve_gate(common_count, reference_count, repro_only_count,
               repro_only_ratio_max):
    """Return coverage metrics and whether a curve has enough support."""
    if reference_count <= 0:
        return False, 0.0, float("inf")
    coverage = common_count / float(reference_count)
    repro_only_ratio = repro_only_count / float(reference_count)
    passed = (common_count >= CURVE_COMMON_COLUMNS_MIN and
              coverage >= CURVE_REFERENCE_COVERAGE_MIN and
              repro_only_ratio <= repro_only_ratio_max)
    return passed, coverage, repro_only_ratio


def label_clusters(image):
    """Dark-pixel clusters of the x tick label region (rows >= 735).

    Returns a list of (start_col, end_col, centroid_x) tuples, left to right.
    """
    region = image[735:, :]
    dark = region.mean(axis=2) < 128
    col_counts = dark.sum(axis=0)
    cols = np.flatnonzero(col_counts > 0)
    if cols.size == 0:
        return []
    groups = []
    start = prev = cols[0]
    for c in cols[1:]:
        if c - prev > 30:
            groups.append((start, prev))
            start = c
        prev = c
    groups.append((start, prev))
    clusters = []
    for lo, hi in groups:
        weights = col_counts[lo:hi + 1].astype(float)
        positions = np.arange(lo, hi + 1, dtype=float)
        centroid = float((positions * weights).sum() / weights.sum())
        clusters.append((int(lo), int(hi), centroid))
    return clusters


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    repro_path = argv[0] if len(argv) > 0 else DEFAULT_REPRO
    ref_path = argv[1] if len(argv) > 1 else DEFAULT_REF

    print("repro: {0}".format(os.path.abspath(repro_path)))
    print("ref  : {0}".format(os.path.abspath(ref_path)))
    print("")

    results = []

    repro = load_image(repro_path)
    ref = load_image(ref_path)

    # --- 1. canvas size ---------------------------------------------------
    ok_size = repro.shape == ref.shape
    same_canvas = repro.shape[:2] == CANVAS
    print("[{0}] canvas size: repro {1} vs ref {2} (expect {3})".format(
        "PASS" if (ok_size and same_canvas) else "FAIL",
        repro.shape[:2], ref.shape[:2], CANVAS))
    results.append(ok_size and same_canvas)
    if not results[-1]:
        # Every later detector uses survey coordinates for this exact canvas.
        # A size mismatch is a complete comparison failure, not an invitation
        # to index fixed coordinates in an incompatible image.
        print("")
        print("checks passed: 0/1")
        print("RESULT: FAIL")
        return 1

    # --- 2. pure-blue pixel count -----------------------------------------
    count_r = int(exact_mask(repro, BLUE).sum())
    count_f = int(exact_mask(ref, BLUE).sum())
    ok_blue = abs(count_r - BLUE_PURE_TARGET) <= PURE_TOLERANCE * BLUE_PURE_TARGET
    print("[{0}] blue pure pixels: repro={1} ref={2} target={3} (+-{4:.0%})"
          "  dev={5:+.1%}".format(
              "PASS" if ok_blue else "FAIL", count_r, count_f,
              BLUE_PURE_TARGET, PURE_TOLERANCE,
              (count_r - BLUE_PURE_TARGET) / float(BLUE_PURE_TARGET)))
    results.append(ok_blue)

    # --- 3. orange must be absent in the repro ----------------------------
    orange_r = exact_mask(repro, ORANGE)
    orange_r[LABEL_REGION_ROW:, :] = False  # rows < 740 only
    n_orange = int(orange_r.sum())
    ok_orange = n_orange == 0
    print("[{0}] orange pixels in repro (rows<740): {1} (must be 0)".format(
        "PASS" if ok_orange else "FAIL", n_orange))
    results.append(ok_orange)

    # --- 4. blue centerline RMSE / p95 ------------------------------------
    blue_r = curve_centers(repro, BLUE)
    blue_f = curve_centers(ref, BLUE)
    cols, devs = center_deviation(blue_r, blue_f)
    only_r = len(blue_r) - len(cols)
    only_f = len(blue_f) - len(cols)
    support_ok, coverage, extra_ratio = curve_gate(
        len(cols), len(blue_f), only_r, BLUE_REPRO_ONLY_RATIO_MAX)
    ok_blue_center = False
    if devs.size:
        rmse = float(np.sqrt((devs ** 2).mean()))
        p95 = float(np.percentile(np.abs(devs), 95))
        ok_blue_center = (rmse <= CENTER_RMSE_MAX and
                          p95 <= CENTER_P95_MAX and support_ok)
        print("[{0}] blue centerline: RMSE={1:.3f}px (<= {2})  p95={3:.3f}px"
              " (<= {4})  columns={5}; coverage={6:.1%} (>= {7:.0%});"
              " repro-only={8} ({9:.1%}, <= {10:.0%}); ref-only={11}".format(
                  "PASS" if ok_blue_center else "FAIL", rmse,
                  CENTER_RMSE_MAX, p95, CENTER_P95_MAX, len(cols),
                  coverage, CURVE_REFERENCE_COVERAGE_MIN, only_r,
                  extra_ratio, BLUE_REPRO_ONLY_RATIO_MAX, only_f))
    else:
        print("[FAIL] blue centerline: no common columns")
    results.append(ok_blue_center)

    # --- 5. red centerline RMSE / p95 (reference columns only) ------------
    red_r = curve_centers(repro, RED)
    red_f = curve_centers(ref, RED)
    cols, devs = center_deviation(red_r, red_f, restrict=set(red_f))
    only_r = len(red_r) - len(cols)
    only_f = len(red_f) - len(cols)
    support_ok, coverage, extra_ratio = curve_gate(
        len(cols), len(red_f), only_r, RED_REPRO_ONLY_RATIO_MAX)
    ok_red_center = False
    if devs.size:
        rmse = float(np.sqrt((devs ** 2).mean()))
        p95 = float(np.percentile(np.abs(devs), 95))
        ok_red_center = (rmse <= CENTER_RMSE_MAX and
                         p95 <= CENTER_P95_MAX and support_ok)
        print("[{0}] red  centerline: RMSE={1:.3f}px (<= {2})  p95={3:.3f}px"
              " (<= {4})  columns={5} of {6}; coverage={7:.1%} (>= {8:.0%});"
              " repro-only={9} ({10:.1%}, <= {11:.0%}); ref-only={12}".format(
                  "PASS" if ok_red_center else "FAIL", rmse,
                  CENTER_RMSE_MAX, p95, CENTER_P95_MAX, len(cols),
                  len(red_f), coverage, CURVE_REFERENCE_COVERAGE_MIN,
                  only_r, extra_ratio, RED_REPRO_ONLY_RATIO_MAX, only_f))
    else:
        print("[FAIL] red centerline: no common columns")
    results.append(ok_red_center)

    # --- 6. grid line centers ---------------------------------------------
    grid_r = detect_grid(repro)
    grid_f = detect_grid(ref)
    ok_grid = True
    for tag in ("major_v", "minor_v", "major_h", "minor_h"):
        list_r, list_f = grid_r[tag], grid_f[tag]
        worst = max_match_dev(list_r, list_f)
        if worst is None:
            ok_grid = False
            print("[FAIL] grid {0}: line count mismatch (repro {1} vs"
                  " ref {2})".format(tag, len(list_r), len(list_f)))
            if os.environ.get("COMPARE_DEBUG"):
                print("       repro: {0}".format(
                    ["%.2f" % c for c in list_r]))
                print("       ref  : {0}".format(
                    ["%.2f" % c for c in list_f]))
            continue
        ok = worst <= GRID_CENTER_MAX
        ok_grid = ok_grid and ok
        print("[{0}] grid {1:8s}: {2} lines, max center dev={3:.2f}px"
              " (<= {4})".format("PASS" if ok else "FAIL", tag,
                                 len(list_r), worst, GRID_CENTER_MAX))
        if not ok and os.environ.get("COMPARE_DEBUG"):
            print("       repro: {0}".format(["%.2f" % c for c in list_r]))
            print("       ref  : {0}".format(["%.2f" % c for c in list_f]))
    results.append(ok_grid)

    # --- 7. x tick label clusters -----------------------------------------
    clusters_r = label_clusters(repro)
    clusters_f = label_clusters(ref)
    ok_clusters = len(clusters_r) == NUM_CLUSTERS == len(clusters_f)
    if ok_clusters:
        devs = [abs(a[2] - b[2]) for a, b in zip(clusters_r, clusters_f)]
        worst = max(devs)
        ok_clusters = worst <= CLUSTER_CENTER_MAX
        print("[{0}] label clusters: {1}, max center dev={2:.2f}px (<= {3})"
              .format("PASS" if ok_clusters else "FAIL", len(clusters_r),
                      worst, CLUSTER_CENTER_MAX))
    else:
        print("[FAIL] label clusters: repro {0} vs ref {1} (expect {2})"
              .format(len(clusters_r), len(clusters_f), NUM_CLUSTERS))
        print("       repro spans: {0}".format(
            [(c[0], c[1]) for c in clusters_r]))
        print("       ref   spans: {0}".format(
            [(c[0], c[1]) for c in clusters_f]))
    results.append(ok_clusters)

    # --- 8. background spot checks ----------------------------------------
    white_r = np.all(repro == 255, axis=2)
    white_f = np.all(ref == 255, axis=2)
    frac_r = float(white_r.mean())
    frac_f = float(white_f.mean())
    bad_spots = []
    for row, col in SPOT_POINTS:
        if white_f[row, col] and not white_r[row, col]:
            bad_spots.append((row, col))
    ok_bg = (frac_r >= WHITE_FRAC_MIN and
             abs(frac_r - frac_f) <= WHITE_FRAC_DIFF and not bad_spots)
    print("[{0}] background: white frac repro={1:.4f} ref={2:.4f}"
          " (min {3}, diff <= {4}); bad spots={5}".format(
              "PASS" if ok_bg else "FAIL", frac_r, frac_f, WHITE_FRAC_MIN,
              WHITE_FRAC_DIFF, bad_spots))
    results.append(ok_bg)

    # --- verdict ----------------------------------------------------------
    passed = all(results)
    print("")
    print("checks passed: {0}/{1}".format(sum(1 for r in results if r),
                                          len(results)))
    print("RESULT: {0}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Negative-case tests for the objective image comparator."""

from __future__ import absolute_import

import contextlib
import io
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

import compare


class CompareSafetyTests(unittest.TestCase):
    def _save(self, directory, name, pixels):
        path = os.path.join(directory, name)
        Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)
        return path

    def _run(self, repro_path, ref_path=compare.DEFAULT_REF):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = compare.main([repro_path, ref_path])
        return code, output.getvalue()

    def _default_rgb(self):
        return compare.load_image(compare.DEFAULT_REPRO).astype(np.uint8)

    def _keep_red_center_columns(self, image, columns):
        result = image.copy()
        red = compare.exact_mask(result, compare.RED)
        keep = np.zeros(result.shape[1], dtype=bool)
        keep[list(columns)] = True
        red[:, keep] = False
        result[red] = 255
        return result

    def test_red_curve_with_only_one_common_column_fails_coverage_gate(self):
        image = self._default_rgb()
        ref_centers = compare.curve_centers(compare.load_image(compare.DEFAULT_REF),
                                             compare.RED)
        repro_centers = compare.curve_centers(image, compare.RED)
        common = sorted(set(ref_centers) & set(repro_centers))
        image = self._keep_red_center_columns(image, [common[len(common) // 2]])

        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, "one-column.png", image)
            code, output = self._run(path)

        self.assertEqual(code, 1)
        self.assertIn("coverage", output.lower())

    def test_red_curve_missing_half_of_reference_columns_fails(self):
        image = self._default_rgb()
        ref_centers = compare.curve_centers(compare.load_image(compare.DEFAULT_REF),
                                             compare.RED)
        repro_centers = compare.curve_centers(image, compare.RED)
        common = sorted(set(ref_centers) & set(repro_centers))
        image = self._keep_red_center_columns(image, common[::2])

        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, "half-red.png", image)
            code, output = self._run(path)

        self.assertEqual(code, 1)
        self.assertIn("coverage", output.lower())

    def test_unrelated_extra_red_columns_fail_repro_only_gate(self):
        image = self._default_rgb()
        ref_centers = compare.curve_centers(compare.load_image(compare.DEFAULT_REF),
                                             compare.RED)
        repro_centers = compare.curve_centers(image, compare.RED)
        extra_columns = sorted(set(range(image.shape[1])) - set(repro_centers)
                               - set(ref_centers))
        red = compare.exact_mask(image, compare.RED)
        for column in extra_columns:
            image[red[:, column], column] = 255
        image[100, extra_columns] = np.array(compare.RED, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, "extra-red.png", image)
            code, output = self._run(path)

        self.assertEqual(code, 1)
        self.assertIn("repro-only", output.lower())

    def test_too_small_canvas_returns_failure_without_index_error(self):
        tiny = np.full((10, 10, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, "tiny.png", tiny)
            code, output = self._run(path)

        self.assertEqual(code, 1)
        self.assertIn("canvas size", output.lower())

    def test_grid_centers_are_matched_one_to_one(self):
        # Nearest-neighbour reuse would match both refs to repro center 0 and
        # incorrectly return 1.  Ordered one-to-one matching must expose 99.
        self.assertEqual(compare.max_match_dev([0.0, 1.0], [0.0, 100.0]), 99.0)


if __name__ == "__main__":
    unittest.main()

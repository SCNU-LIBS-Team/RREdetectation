import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

plt.show = lambda *args, **kwargs: None

import Identification_Matrix as im


class IdentificationMatrixSaveTests(unittest.TestCase):
    def test_generate_identification_matrix_prints_average_precision_and_recall(self):
        contents = pd.DataFrame(
            [
                ["sample_1", 1, 0, 0, 0, 0, 0],
                ["sample_2", 1, 1, 0, 0, 0, 0],
            ],
            columns=["Sample", "Ce", "La", "meta1", "meta2", "meta3", "meta4"],
        )
        predictions = pd.DataFrame(
            [
                ["sample_1", 1, 1, 0, 0, 0, 0],
                ["sample_2", 0, 1, 0, 0, 0, 0],
            ],
            columns=["Sample", "Ce", "La", "meta1", "meta2", "meta3", "meta4"],
        )

        output = StringIO()
        with redirect_stdout(output):
            im.generate_identification_matrix(
                contents,
                predictions,
                show_plot=False,
                show_metrics_plot=True,
            )

        printed = output.getvalue()
        self.assertIn("全部元素平均正确率 (Average Precision): 75.00%", printed)
        self.assertIn("全部元素平均检出率 (Average Recall): 75.00%", printed)

    def test_generate_identification_matrix_saves_both_plots_at_3000_dpi(self):
        contents = pd.DataFrame(
            [
                ["sample_1", 1, 0, 0, 0, 0, 0],
                ["sample_2", 0, 1, 0, 0, 0, 0],
            ],
            columns=["Sample", "Ce", "La", "meta1", "meta2", "meta3", "meta4"],
        )
        predictions = pd.DataFrame(
            [
                ["sample_1", 1, 1, 0, 0, 0, 0],
                ["sample_2", 0, 1, 0, 0, 0, 0],
            ],
            columns=["Sample", "Ce", "La", "meta1", "meta2", "meta3", "meta4"],
        )

        save_calls = []
        original_savefig = matplotlib.figure.Figure.savefig

        def record_savefig(fig, path, *args, **kwargs):
            save_calls.append((Path(path).name, kwargs.get("dpi"), kwargs.get("bbox_inches")))

        matplotlib.figure.Figure.savefig = record_savefig
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with redirect_stdout(StringIO()):
                    im.generate_identification_matrix(
                        contents,
                        predictions,
                        show_plot=True,
                        show_metrics_plot=True,
                        save_dir=tmpdir,
                    )
        finally:
            matplotlib.figure.Figure.savefig = original_savefig

        self.assertEqual(
            save_calls,
            [
                ("identification_matrix.png", 3000, "tight"),
                ("element_metrics.png", 3000, "tight"),
            ],
        )


if __name__ == "__main__":
    unittest.main()

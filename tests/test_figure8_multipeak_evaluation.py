from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "Paper" / "Plot" / "figure8" / "evaluate_multipeak_fit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("figure8_evaluator", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_present_rare_earths_uses_positive_atomic_percent(tmpdir):
    evaluator = _load_module()
    tmp_path = Path(str(tmpdir))
    contents_path = tmp_path / "Randomrareearth_contents.csv"
    pd.DataFrame(
        [
            {
                "file_name": "sample_a.csv",
                "Tm_at%": 5.0,
                "Ce_at%": 0.0,
                "Y_at%": np.nan,
                "Te_eV": 9500.0,
            },
            {
                "file_name": "sample_b.csv",
                "Tm_at%": 0.0,
                "Ce_at%": 0.25,
                "Y_at%": -0.1,
                "Te_eV": 9000.0,
            },
        ]
    ).to_csv(contents_path, index=False)

    present = evaluator._load_present_rare_earths(contents_path)

    assert present == {
        "sample_a": {"TM"},
        "sample_b": {"CE"},
    }


def test_default_input_resolution_uses_only_spectra_listed_in_contents(tmpdir, monkeypatch):
    evaluator = _load_module()
    tmp_path = Path(str(tmpdir))
    monkeypatch.setattr(evaluator, "SCRIPT_DIR", tmp_path)
    (tmp_path / "sample_a.csv").touch()
    (tmp_path / "sample_b.csv").touch()
    (tmp_path / "Randomrareearth_contents.csv").touch()
    (tmp_path / "contents.csv").touch()

    inputs = evaluator._resolve_inputs([], {"sample_b", "sample_a"})

    assert [path.name for path in inputs] == ["sample_a.csv", "sample_b.csv"]


def test_default_input_resolution_rejects_spectrum_missing_from_directory(tmpdir, monkeypatch):
    evaluator = _load_module()
    tmp_path = Path(str(tmpdir))
    monkeypatch.setattr(evaluator, "SCRIPT_DIR", tmp_path)
    (tmp_path / "sample_a.csv").touch()

    try:
        evaluator._resolve_inputs([], {"sample_a", "sample_missing"})
    except FileNotFoundError as error:
        assert "sample_missing.csv" in str(error)
    else:
        raise AssertionError("A contents-table spectrum missing from disk must fail.")


def test_explicit_input_resolution_rejects_file_not_listed_in_contents(tmpdir, monkeypatch):
    evaluator = _load_module()
    tmp_path = Path(str(tmpdir))
    monkeypatch.setattr(evaluator, "SCRIPT_DIR", tmp_path)
    (tmp_path / "contents.csv").touch()

    try:
        evaluator._resolve_inputs(["contents.csv"], {"sample_a"})
    except ValueError as error:
        assert "contents.csv" in str(error)
    else:
        raise AssertionError("An explicit helper/unlisted CSV must be rejected.")


def test_contents_path_keeps_relative_parent_directory(tmpdir, monkeypatch):
    evaluator = _load_module()
    tmp_path = Path(str(tmpdir))
    monkeypatch.chdir(str(tmp_path))

    resolved = evaluator._resolve_contents_path(Path("fixtures/custom_contents.csv"))

    assert resolved == (tmp_path / "fixtures" / "custom_contents.csv").resolve()


def test_output_rows_keep_only_fitted_rare_earths_present_in_spectrum(tmpdir):
    evaluator = _load_module()
    tmp_path = Path(str(tmpdir))
    input_path = tmp_path / "sample_a.csv"
    spectrum = pd.DataFrame(
        {
            "Wavelength (nm)": [399.9, 400.0, 400.1],
            "Sum(calc)": [0.0, 10.0, 0.0],
            "Tm II (5.0e-2)": [0.0, 8.0, 0.0],
        }
    )
    common = {
        "spectrum_name": "sample_a",
        "fit_interval": (399.9, 400.1),
        "unremoved_interval": (399.8, 400.2),
        "selected_lines": "target line",
        "rmse": 0.5,
        "target_wavelength": 400.0,
    }
    diagnostics = [
        {**common, "particle": "TmII", "fitted_amplitude": 7.5},
        {**common, "particle": "CeII", "fitted_amplitude": 2.0},
        {**common, "particle": "FeI", "fitted_amplitude": 1.0},
    ]

    rows = evaluator._build_output_rows(
        input_path,
        spectrum,
        diagnostics,
        present_rare_earths={"TM"},
    )

    assert len(rows) == 1
    assert rows[0]["光谱名称"] == "sample_a.csv"
    assert rows[0]["粒子种类"] == "TmII"
    assert rows[0]["真实强度"] == 8.0
    assert rows[0]["拟合强度"] == 7.5
    assert np.isclose(rows[0]["RMSE/nm"], 2.5)
    assert rows[0]["强度误差"] == 0.5
    assert rows[0]["强度误差比例(%)"] == 6.25


def test_combined_output_writes_all_rows_to_one_csv(tmpdir):
    evaluator = _load_module()
    tmp_path = Path(str(tmpdir))
    output_path = tmp_path / "evaluation_results" / "multipeak_fit_results.csv"
    rows = [
        {
            "光谱名称": "sample_a.csv",
            "粒子种类": "TmII",
            "拟合区间(nm)": "399.900000-400.100000",
            "未去除旁峰的区间(nm)": "399.800000-400.200000",
            "选线数据": "TmII 400.000000 nm (target)",
            "RMSE": 0.5,
            "RMSE/nm": 2.5,
            "真实强度": 8.0,
            "拟合强度": 7.5,
            "强度误差": 0.5,
            "强度误差比例(%)": 6.25,
        },
        {
            "光谱名称": "sample_b.csv",
            "粒子种类": "CeII",
            "拟合区间(nm)": "410.000000-410.200000",
            "未去除旁峰的区间(nm)": "409.900000-410.300000",
            "选线数据": "CeII 410.100000 nm (target)",
            "RMSE": 0.25,
            "RMSE/nm": 1.25,
            "真实强度": 5.0,
            "拟合强度": 4.9,
            "强度误差": 0.1,
            "强度误差比例(%)": 2.0,
        },
    ]

    evaluator._write_combined_output(output_path, rows)

    written = pd.read_csv(output_path, encoding="utf-8-sig")
    assert output_path.is_file()
    assert list(written.columns) == evaluator.OUTPUT_COLUMNS
    assert evaluator.OUTPUT_COLUMNS.index("RMSE/nm") == evaluator.OUTPUT_COLUMNS.index("RMSE") + 1
    assert evaluator.OUTPUT_COLUMNS.index("拟合强度") == evaluator.OUTPUT_COLUMNS.index("真实强度") + 1
    assert written.iloc[:-2]["光谱名称"].tolist() == ["sample_a.csv", "sample_b.csv"]
    assert written.iloc[:-2]["真实强度"].tolist() == [8.0, 5.0]
    assert written.iloc[:-2]["拟合强度"].tolist() == [7.5, 4.9]
    assert written.iloc[-2]["光谱名称"] == "强度误差比例平均值"
    assert np.isclose(written.iloc[-2]["强度误差比例(%)"], 4.125)
    assert pd.isna(written.iloc[-2]["RMSE/nm"])
    assert written.iloc[-1]["光谱名称"] == "RMSE/nm平均值"
    assert np.isclose(written.iloc[-1]["RMSE/nm"], 1.875)
    assert pd.isna(written.iloc[-1]["强度误差比例(%)"])


def test_combined_output_average_ignores_non_finite_error_percentages(tmpdir):
    evaluator = _load_module()
    output_path = Path(str(tmpdir)) / "multipeak_fit_results.csv"
    rows = [
        {"光谱名称": "finite.csv", "RMSE/nm": 2.0, "强度误差比例(%)": 10.0},
        {"光谱名称": "infinite.csv", "RMSE/nm": np.inf, "强度误差比例(%)": np.inf},
        {"光谱名称": "missing.csv", "RMSE/nm": np.nan, "强度误差比例(%)": np.nan},
    ]

    evaluator._write_combined_output(output_path, rows)

    written = pd.read_csv(output_path, encoding="utf-8-sig")
    assert written.iloc[-2]["光谱名称"] == "强度误差比例平均值"
    assert written.iloc[-2]["强度误差比例(%)"] == 10.0
    assert written.iloc[-1]["光谱名称"] == "RMSE/nm平均值"
    assert written.iloc[-1]["RMSE/nm"] == 2.0

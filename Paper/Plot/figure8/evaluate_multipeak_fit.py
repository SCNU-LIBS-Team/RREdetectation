"""Run the real Elements_detectation workflow and export peak-fit errors.

The script deliberately reuses the ``if __name__ == '__main__'`` body from
``Elements_detectation.py``.  It only wraps the existing fitter long enough to
collect diagnostics; no detection, line-selection, side-peak removal, or
multi-peak fitting logic is copied here.

By default every spectrum listed by ``Randomrareearth_contents.csv`` and found
directly inside this directory is processed.  Only fitted rare-earth elements
whose generated abundance is positive are exported, and all spectra are
combined into one CSV under ``evaluation_results``.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

# This is a batch evaluator.  Disabling GUI windows does not change fitting.
matplotlib.use("Agg")

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DETECTOR_PATH = REPO_ROOT / "Elements_detectation.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evaluation_results"
DEFAULT_CONTENTS_PATH = SCRIPT_DIR / "Randomrareearth_contents.csv"
DEFAULT_OUTPUT_NAME = "multipeak_fit_results.csv"
RARE_EARTH_ELEMENTS = {
    "Y",
    "LA",
    "CE",
    "PR",
    "ND",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
}

OUTPUT_COLUMNS = [
    "光谱名称",
    "粒子种类",
    "拟合区间(nm)",
    "未去除旁峰的区间(nm)",
    "选线数据",
    "RMSE",
    "RMSE/nm",
    "真实强度",
    "拟合强度",
    "强度误差",
    "强度误差比例(%)",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Elements_detectation 的真实流程评价多峰拟合峰高。",
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="可选：仅处理指定 CSV（可写文件名或不带 .csv 的文件名）；默认处理含量表中列出的全部光谱。",
    )
    parser.add_argument(
        "--contents-file",
        type=Path,
        default=DEFAULT_CONTENTS_PATH,
        help="稀土真实含量表；默认是本目录下的 Randomrareearth_contents.csv。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="结果目录；默认是本目录下的 evaluation_results。",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"合并结果文件名；默认是 {DEFAULT_OUTPUT_NAME}。",
    )
    return parser.parse_args()


def _load_present_rare_earths(contents_path: Path) -> dict[str, set[str]]:
    if not contents_path.is_file():
        raise FileNotFoundError(f"找不到稀土真实含量表：{contents_path}")

    contents = pd.read_csv(contents_path, header=0, encoding="utf-8-sig")
    if "file_name" not in contents.columns:
        raise ValueError(f"稀土真实含量表缺少 file_name 列：{contents_path}")

    element_columns: dict[str, str] = {}
    for column in contents.columns:
        match = re.fullmatch(r"([A-Za-z]{1,2})_at%", str(column).strip())
        if match is None:
            continue
        element = match.group(1).upper()
        if element in RARE_EARTH_ELEMENTS:
            element_columns[element] = str(column)

    if not element_columns:
        raise ValueError(f"稀土真实含量表中没有可用的 <元素>_at% 列：{contents_path}")

    present_by_spectrum: dict[str, set[str]] = {}
    for _, row in contents.iterrows():
        file_name = str(row["file_name"]).strip()
        if not file_name or file_name.lower() == "nan":
            continue
        spectrum_name = Path(file_name).stem
        present = present_by_spectrum.setdefault(spectrum_name, set())
        for element, column in element_columns.items():
            abundance = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(abundance) and float(abundance) > 0.0:
                present.add(element)

    if not present_by_spectrum:
        raise ValueError(f"稀土真实含量表中没有有效的光谱记录：{contents_path}")
    return present_by_spectrum


def _resolve_inputs(
    filenames: list[str],
    available_spectrum_names: set[str] | None = None,
) -> list[Path]:
    if not filenames:
        if available_spectrum_names is None:
            inputs = sorted(path for path in SCRIPT_DIR.glob("*.csv") if path.is_file())
        else:
            inputs = [
                (SCRIPT_DIR / f"{spectrum_name}.csv").resolve()
                for spectrum_name in sorted(available_spectrum_names)
            ]
    else:
        inputs = []
        for filename in filenames:
            requested = Path(filename)
            if requested.suffix.lower() != ".csv":
                requested = requested.with_suffix(".csv")
            if not requested.is_absolute():
                requested = SCRIPT_DIR / requested.name
            inputs.append(requested.resolve())

        if available_spectrum_names is not None:
            unlisted = [path.name for path in inputs if path.stem not in available_spectrum_names]
            if unlisted:
                raise ValueError(
                    "指定文件未列在稀土真实含量表中：" + ", ".join(unlisted)
                )

    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("找不到待测光谱：" + ", ".join(missing))
    if not inputs:
        raise FileNotFoundError(f"{SCRIPT_DIR} 下没有待测 CSV。")
    return inputs


def _resolve_contents_path(contents_path: Path) -> Path:
    if contents_path.is_absolute():
        return contents_path.resolve()
    return (Path.cwd() / contents_path).resolve()


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
        return False
    comparator = test.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def _compile_detector_main() -> Any:
    """Compile the original main block without copying its orchestration."""
    source = DETECTOR_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(DETECTOR_PATH))
    main_guard = next(
        (node for node in tree.body if isinstance(node, ast.If) and _is_main_guard(node)),
        None,
    )
    if main_guard is None:
        raise RuntimeError("Elements_detectation.py 中未找到主流程入口。")

    main_body = ast.Module(body=main_guard.body, type_ignores=[])
    ast.fix_missing_locations(main_body)
    return compile(main_body, str(DETECTOR_PATH), "exec")


def _canonical_particle_name(value: object) -> str:
    # Simulation headers look like "Tm II (4.8e-2)" while the detector uses
    # "TmII".  The concentration suffix is metadata, not part of the species.
    label = str(value).split("(", 1)[0]
    return re.sub(r"[^A-Za-z0-9]", "", label).upper()


def _particle_element_symbol(value: object) -> str:
    canonical = _canonical_particle_name(value)
    for symbol in sorted(RARE_EARTH_ELEMENTS, key=len, reverse=True):
        ion_stage = canonical[len(symbol):]
        if canonical.startswith(symbol) and (not ion_stage or set(ion_stage) <= {"I", "V"}):
            return symbol
    return ""


def _is_rare_earth_particle(value: object) -> bool:
    return _particle_element_symbol(value) in RARE_EARTH_ELEMENTS


def _particle_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        _canonical_particle_name(column): str(column)
        for column in frame.columns[2:]
    }


def _finite_interval(values: object) -> tuple[float, float] | None:
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.min(array)), float(np.max(array))


def _format_interval(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return ""
    return f"{interval[0]:.6f}-{interval[1]:.6f}"


def _line_particle_name(row: pd.Series) -> str:
    line_type = str(row.get("LineType", "")).split("(", 1)[0].strip()
    if line_type and line_type.lower() != "nan":
        return re.sub(r"\s+", "", line_type)
    return str(row.get("Element", "")).strip()


def _component_particle_names(
    peak_mu: object,
    target_particle: str,
    target_wavelength: float,
    strongest_line_rows: object,
    line_wavelength_column: object,
) -> list[str]:
    selected = np.asarray(peak_mu, dtype=float).reshape(-1)

    matrix_lines: list[tuple[str, float]] = []
    if isinstance(strongest_line_rows, pd.DataFrame) and not strongest_line_rows.empty:
        wavelength_column = str(line_wavelength_column)
        if wavelength_column in strongest_line_rows.columns:
            for _, row in strongest_line_rows.iterrows():
                wavelength = pd.to_numeric(row[wavelength_column], errors="coerce")
                if pd.notna(wavelength):
                    matrix_lines.append((_line_particle_name(row), float(wavelength)))

    particle_names: list[str] = []
    target_used = False
    used_matrix_rows: set[int] = set()
    for wavelength in selected:
        if not np.isfinite(wavelength):
            particle_names.append("")
            continue
        if not target_used and np.isclose(wavelength, target_wavelength, rtol=0.0, atol=1e-6):
            particle_names.append(target_particle)
            target_used = True
            continue

        matched_index = next(
            (
                index
                for index, (_, line_wavelength) in enumerate(matrix_lines)
                if index not in used_matrix_rows
                and np.isclose(wavelength, line_wavelength, rtol=0.0, atol=1e-6)
            ),
            None,
        )
        if matched_index is None:
            particle_names.append(target_particle)
        else:
            used_matrix_rows.add(matched_index)
            line_particle, _ = matrix_lines[matched_index]
            particle_names.append(line_particle)

    return particle_names


def _format_selected_lines(
    peak_mu: object,
    particle_names: list[str],
    target_particle: str,
    target_wavelength: float,
) -> str:
    selected = np.asarray(peak_mu, dtype=float).reshape(-1)
    descriptions: list[str] = []
    target_used = False
    for index, wavelength in enumerate(selected):
        if not np.isfinite(wavelength):
            continue
        particle = particle_names[index] if index < len(particle_names) else ""
        is_target = (
            not target_used
            and particle == target_particle
            and np.isclose(wavelength, target_wavelength, rtol=0.0, atol=1e-6)
        )
        descriptions.append(
            f"{particle} {wavelength:.6f} nm" + (" (target)" if is_target else "")
        )
        if is_target:
            target_used = True

    return "; ".join(descriptions)


def _nearest_true_peak_intensity(
    spectrum: pd.DataFrame,
    particle_column: str,
    target_wavelength: float,
    fit_interval: tuple[float, float],
    max_wavelength_difference: float = 0.2,
) -> float | None:
    """Return the particle-column local peak nearest the selected line."""
    wavelength = pd.to_numeric(spectrum.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    intensity = (
        pd.to_numeric(spectrum[particle_column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(intensity)
        & (wavelength >= fit_interval[0])
        & (wavelength <= fit_interval[1])
    )
    local_wavelength = wavelength[valid]
    local_intensity = intensity[valid]
    if local_wavelength.size == 0:
        return None

    order = np.argsort(local_wavelength)
    local_wavelength = local_wavelength[order]
    local_intensity = local_intensity[order]

    if local_intensity.size >= 3:
        center = local_intensity[1:-1]
        peak_indices = np.flatnonzero(
            (center >= local_intensity[:-2])
            & (center >= local_intensity[2:])
            & ((center > local_intensity[:-2]) | (center > local_intensity[2:]))
            & (center > 0.0)
        ) + 1
    else:
        peak_indices = np.array([], dtype=int)

    # A target at an extracted-window edge is unusual but still comparable.
    edge_indices: list[int] = []
    if local_intensity.size == 1 and local_intensity[0] > 0.0:
        edge_indices.append(0)
    elif local_intensity.size >= 2:
        if local_intensity[0] > local_intensity[1] and local_intensity[0] > 0.0:
            edge_indices.append(0)
        if local_intensity[-1] > local_intensity[-2] and local_intensity[-1] > 0.0:
            edge_indices.append(local_intensity.size - 1)
    if edge_indices:
        peak_indices = np.unique(np.concatenate([peak_indices, np.asarray(edge_indices, dtype=int)]))

    if peak_indices.size == 0:
        return None

    wavelength_difference = np.abs(local_wavelength[peak_indices] - float(target_wavelength))
    nearest_position = int(np.argmin(wavelength_difference))
    if wavelength_difference[nearest_position] > float(max_wavelength_difference):
        return None

    true_intensity = float(local_intensity[peak_indices[nearest_position]])
    if not np.isfinite(true_intensity) or true_intensity <= 0.0:
        return None
    return true_intensity


def _capture_fit_diagnostics(fitter: object, caller_frame: Any) -> list[dict[str, object]]:
    local_values = caller_frame.f_locals
    if caller_frame.f_code.co_name != "MultiPeakFit":
        return []

    fitted_params = np.asarray(getattr(fitter, "fitted_params", []), dtype=float)
    if fitted_params.size == 0:
        return []
    fitted_params = np.atleast_2d(fitted_params)

    fit_interval = _finite_interval(getattr(fitter, "wl", []))
    if fit_interval is None:
        return []

    observed = np.asarray(getattr(fitter, "rel_int", []), dtype=float).reshape(-1)
    predicted = np.asarray(getattr(fitter, "total_fit", []), dtype=float).reshape(-1)
    if observed.shape != predicted.shape:
        rmse = np.nan
    else:
        valid_rmse = np.isfinite(observed) & np.isfinite(predicted)
        if not np.any(valid_rmse):
            rmse = np.nan
        else:
            rmse = float(np.sqrt(np.mean((observed[valid_rmse] - predicted[valid_rmse]) ** 2)))

    particle = str(local_values.get("element_name", ""))
    target_wavelength = float(local_values.get("wl_value", np.nan))
    component_particles = _component_particle_names(
        fitted_params[:, 1],
        particle,
        target_wavelength,
        local_values.get("strongest_line_rows"),
        local_values.get("line_wl_col"),
    )
    selected_lines = _format_selected_lines(
        fitted_params[:, 1],
        component_particles,
        particle,
        target_wavelength,
    )

    common_values = {
        "fit_interval": fit_interval,
        "unremoved_interval": _finite_interval(local_values.get("extra_segment_wl", [])),
        "selected_lines": selected_lines,
        "rmse": rmse,
    }
    return [
        {
            **common_values,
            "particle": component_particles[index],
            "target_wavelength": float(mu),
            "fitted_amplitude": float(amplitude),
        }
        for index, (amplitude, mu, _sigma) in enumerate(fitted_params)
    ]


def _run_real_workflow(input_paths: list[Path]) -> list[dict[str, object]]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import Elements_detectation as detector

    main_code = _compile_detector_main()
    diagnostics: list[dict[str, object]] = []

    original_fit = detector.GaussMultiPeakFitter.fit

    def captured_fit(fitter: object, *args: object, **kwargs: object) -> object:
        result = original_fit(fitter, *args, **kwargs)
        current_frame = inspect.currentframe()
        caller_frame = None if current_frame is None else current_frame.f_back
        if caller_frame is not None:
            captured_diagnostics = _capture_fit_diagnostics(fitter, caller_frame)
            for diagnostic in captured_diagnostics:
                diagnostic["spectrum_name"] = str(getattr(detector, "I_element_name", ""))
            diagnostics.extend(captured_diagnostics)
        del current_frame
        return result

    detector.GaussMultiPeakFitter.fit = captured_fit

    previous_values = {
        name: detector.__dict__.get(name)
        for name in (
            "target_path",
            "I_file_list",
            "I_elements_list",
            "target_files",
            "files_to_process",
            "plotbotton",
            "save2csvbotton",
            "printbotton",
            "plottarget",
        )
    }

    try:
        detector.target_path = str(SCRIPT_DIR)
        detector.I_file_list = [str(path) for path in input_paths]
        detector.I_elements_list = [path.stem for path in input_paths]
        detector.target_files = [path.stem for path in input_paths]
        detector.files_to_process = [path.stem for path in input_paths]
        detector.plotbotton = False
        detector.save2csvbotton = False
        detector.printbotton = False
        detector.plottarget = "__FIGURE8_BATCH_NO_PLOT__"
        exec(main_code, detector.__dict__)
    finally:
        detector.GaussMultiPeakFitter.fit = original_fit
        for name, value in previous_values.items():
            detector.__dict__[name] = value

    return diagnostics


def _build_output_rows(
    input_path: Path,
    spectrum: pd.DataFrame,
    diagnostics: list[dict[str, object]],
    present_rare_earths: set[str],
) -> list[dict[str, object]]:
    columns_by_particle = _particle_columns(spectrum)
    output_rows: list[dict[str, object]] = []

    for diagnostic in diagnostics:
        if str(diagnostic["spectrum_name"]) != input_path.stem:
            continue

        particle = str(diagnostic["particle"])
        particle_element = _particle_element_symbol(particle)
        if not particle_element or particle_element not in present_rare_earths:
            continue

        particle_column = columns_by_particle.get(_canonical_particle_name(particle))
        fit_interval = diagnostic["fit_interval"]
        if not isinstance(fit_interval, tuple):
            continue

        rmse = float(diagnostic["rmse"])
        fit_window_length = abs(float(fit_interval[1]) - float(fit_interval[0]))
        rmse_per_nm = (
            rmse / fit_window_length
            if np.isfinite(rmse) and np.isfinite(fit_window_length) and fit_window_length > 0.0
            else np.nan
        )

        # A particle omitted from the simulation CSV has zero true
        # contribution.  Likewise, if its column exists but has no local peak
        # corresponding to this fitted line, the true peak height is zero.
        if particle_column is None:
            true_intensity = 0.0
        else:
            matched_true_intensity = _nearest_true_peak_intensity(
                spectrum,
                particle_column,
                float(diagnostic["target_wavelength"]),
                fit_interval,
            )
            true_intensity = 0.0 if matched_true_intensity is None else matched_true_intensity

        fitted_amplitude = float(diagnostic["fitted_amplitude"])
        intensity_error = (
            abs(fitted_amplitude - true_intensity)
            if np.isfinite(fitted_amplitude)
            else np.nan
        )
        if not np.isfinite(intensity_error):
            intensity_error_percent = np.nan
        elif true_intensity > 0.0:
            intensity_error_percent = intensity_error / abs(true_intensity) * 100.0
        elif intensity_error == 0.0:
            intensity_error_percent = 0.0
        else:
            intensity_error_percent = np.inf

        output_rows.append(
            {
                "光谱名称": input_path.name,
                "粒子种类": particle,
                "拟合区间(nm)": _format_interval(fit_interval),
                "未去除旁峰的区间(nm)": _format_interval(diagnostic["unremoved_interval"]),
                "选线数据": str(diagnostic["selected_lines"]),
                "RMSE": rmse,
                "RMSE/nm": float(rmse_per_nm),
                "真实强度": float(true_intensity),
                "拟合强度": float(fitted_amplitude),
                "强度误差": float(intensity_error),
                "强度误差比例(%)": float(intensity_error_percent),
            }
        )

    return output_rows


def _write_combined_output(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    rmse_per_nm_values = pd.to_numeric(
        output["RMSE/nm"],
        errors="coerce",
    )
    finite_rmse_per_nm_values = rmse_per_nm_values[np.isfinite(rmse_per_nm_values)]
    error_percentages = pd.to_numeric(
        output["强度误差比例(%)"],
        errors="coerce",
    )
    finite_error_percentages = error_percentages[np.isfinite(error_percentages)]

    error_summary_row: dict[str, object] = {column: np.nan for column in OUTPUT_COLUMNS}
    error_summary_row["光谱名称"] = "强度误差比例平均值"
    if not finite_error_percentages.empty:
        error_summary_row["强度误差比例(%)"] = float(finite_error_percentages.mean())

    rmse_summary_row: dict[str, object] = {column: np.nan for column in OUTPUT_COLUMNS}
    rmse_summary_row["光谱名称"] = "RMSE/nm平均值"
    if not finite_rmse_per_nm_values.empty:
        rmse_summary_row["RMSE/nm"] = float(finite_rmse_per_nm_values.mean())

    output = pd.concat(
        [
            output,
            pd.DataFrame(
                [error_summary_row, rmse_summary_row],
                columns=OUTPUT_COLUMNS,
            ),
        ],
        ignore_index=True,
    )
    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> int:
    args = _parse_args()
    contents_path = _resolve_contents_path(args.contents_file)
    present_by_spectrum = _load_present_rare_earths(contents_path)
    input_paths = _resolve_inputs(args.filenames, set(present_by_spectrum))
    output_dir = args.output_dir.resolve()
    output_path = output_dir / args.output_name

    spectra = {
        input_path.stem: pd.read_csv(input_path, header=0, encoding="utf-8-sig")
        for input_path in input_paths
    }
    diagnostics = _run_real_workflow(input_paths)

    all_rows: list[dict[str, object]] = []
    for input_path in input_paths:
        rows = _build_output_rows(
            input_path,
            spectra[input_path.stem],
            diagnostics,
            present_by_spectrum.get(input_path.stem, set()),
        )
        all_rows.extend(rows)
        print(f"[{input_path.name}] 保留 {len(rows)} 条真实存在稀土元素的多峰拟合记录。")

    _write_combined_output(output_path, all_rows)
    print(f"已合并写入 {len(all_rows)} 条多峰拟合记录：{output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

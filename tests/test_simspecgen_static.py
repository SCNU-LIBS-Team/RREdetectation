from pathlib import Path
import sys

import pandas as pd
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMSPECGEN_ROOT = PROJECT_ROOT / "SimspecGen"
sys.path.insert(0, str(SIMSPECGEN_ROOT))

from simLIBS.simulation import SimulatedLIBS  # noqa: E402


def test_static_mode_keeps_nist_grid_and_does_not_interpolate(monkeypatch):
    expected = pd.DataFrame(
        {
            "wavelength": [199.36, 199.38, 199.41],
            "intensity": [6.869e-50, 1.184e-47, 4.007e-37],
        }
    )

    def retrieve_without_network(instance):
        instance.ion_spectra = pd.DataFrame(
            {
                "Wavelength (nm)": expected["wavelength"],
                "Sum(calc)": expected["intensity"],
            }
        )
        instance._set_spectrum_from_nist_csv()

    def interpolation_must_not_run(*args, **kwargs):
        raise AssertionError("static mode must not remap the NIST wavelength grid")

    monkeypatch.setattr(SimulatedLIBS, "retrieve_data_static", retrieve_without_network)
    monkeypatch.setattr(SimulatedLIBS, "interpolate", interpolation_must_not_run)

    libs = SimulatedLIBS(
        elements=["Fe"],
        percentages=[100.0],
        webscraping="static",
    )

    pd.testing.assert_frame_equal(libs.get_raw_spectrum(), expected)
    pd.testing.assert_frame_equal(libs.get_interpolated_spectrum(), expected)


def test_node_runner_uses_the_same_csv_number_format_as_nist():
    instance = object.__new__(SimulatedLIBS)
    instance.resolution = 3000
    data_source = """
var spectra = {};
var composition = 'Fe:100';
var lines = [];
var levels = {};
var ips = {};
var temp = 1.0;
var eden = 1e17;
var resolution = 180;
var cmeV = 8065.54393734921;
var koeff = 6.043e21;
var dataDopplerArray = [[{label:'Wavelength (nm)',type:'number'}],[200]];
"""
    lte_source = """
function memAvail() { return 1; }
function lte_spectrum_js() {
  return [[
    [{label:'Wavelength (nm)',type:'number'}, {label:'Sum(calc)',type:'number'},
     {label:'Fe I (1.0e+0)',type:'number'}],
    ['199.36', 12.34567, null],
    ['199.38', 0, 0.012345]
  ], [], '', ''];
}
"""

    csv_text = instance._run_nist_lte_javascript(
        data_source,
        lte_source,
        {
            "composition": "Fe:100",
            "temperature": "1.0",
            "electron_density": "1e17",
        },
    )

    assert csv_text == (
        "Wavelength (nm),Sum(calc),Fe I (1.0e+0)\n"
        "199.36,1.235e+1,\n"
        "199.38,0,1.235e-2"
    )


def test_save_to_csv_preserves_complete_nist_csv(tmp_path):
    instance = object.__new__(SimulatedLIBS)
    instance.nist_csv_text = (
        "Wavelength (nm),Sum(calc),Fe I (1.0e+0)\n"
        "199.36,1.235e+1,"
    )
    instance.interpolated_spectrum = pd.DataFrame(
        {"wavelength": [199.36], "intensity": [12.35]}
    )
    output_path = tmp_path / "complete.csv"

    instance.save_to_csv(output_path)

    assert output_path.read_text(encoding="utf-8") == (
        instance.nist_csv_text + "\n"
    )


def test_save_to_csv_can_still_write_legacy_two_column_format(tmp_path):
    instance = object.__new__(SimulatedLIBS)
    instance.interpolated_spectrum = pd.DataFrame(
        {"wavelength": [199.36], "intensity": [12.35]}
    )
    output_path = tmp_path / "legacy.csv"

    instance.save_to_csv(output_path, include_all_columns=False)

    assert output_path.read_text(encoding="utf-8") == (
        "wavelength,intensity\n199.36,12.35\n"
    )


def test_static_requests_do_not_use_the_broken_environment_proxy():
    session = SimulatedLIBS._new_static_session()

    assert session.trust_env is False


def test_static_recalculation_uses_the_same_form_values_as_browser():
    soup = BeautifulSoup(
        """
        <span id="elem1">Fe</span><input id="perc1" name="perc" value="99.75">
        <span id="elem2">Mg</span><input id="perc2" name="perc" value="0.25">
        <input name="temp" value="0.86">
        <input name="eden" value="1e17">
        """,
        "html.parser",
    )

    inputs = SimulatedLIBS._extract_recalculation_inputs(soup)

    assert inputs == {
        "composition": "Fe:99.75;Mg:0.25",
        "temperature": "0.86",
        "electron_density": "1e17",
    }


def test_generator_entry_points_use_static_mode():
    source = (SIMSPECGEN_ROOT / "Gen.py").read_text(encoding="utf-8")

    assert source.count("webscraping='static'") == 3
    assert "webscraping='dynamic'" not in source

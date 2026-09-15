import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
import requests
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import random
import os
from scipy.interpolate import CubicSpline
from concurrent.futures import ThreadPoolExecutor
import math
import urllib3

urllib3.disable_warnings()

BROWSER_TIMEOUT_SECONDS = 420
CHROMEDRIVER_TRANSPORT_TIMEOUT_SECONDS = 540
STATIC_REQUEST_TIMEOUT_SECONDS = 180
STATIC_CALCULATION_TIMEOUT_SECONDS = 420
NIST_LTE_JAVASCRIPT_URL = (
    "https://physics.nist.gov/PhysRefData/ASD/LIBS/js/saha_lte.js"
)
NIST_LTE_RUNNER = Path(__file__).with_name("nist_lte_runner.js")


class CompositionError(Exception):
    pass


def validate_simulated_libs(
    Te: float,
    Ne: float,
    elements: List[str],
    percentages: List[float],
    low_w: int,
    upper_w: int,
    max_ion_charge: int,
):
    """
    Validates the input parameters for simLIBS.

    Raises:
        CompositionError: If the sum of percentages exceeds 100.
        ValueError: If any parameter has an invalid value (negative, wrong order).
    """
    if sum(percentages) > 100:
        raise CompositionError("Sum of element percentages cannot exceed 100%")
    if (
        any(value < 0 for value in [Te, Ne, max_ion_charge])
        or low_w > upper_w
        or len(elements) != len(percentages)
    ):
        raise ValueError(
            "Invalid parameters: negative values, wrong wavelength order, or element/percentage mismatch."
        )


class SimulatedLIBS(object):

    _nist_lte_javascript = None
    _nist_lte_javascript_lock = threading.Lock()

    def __init__(
        self,
        Te: float = 1.0,
        Ne: float = 10**17,
        elements: list[str] = None,
        percentages: list[float] = None,
        resolution: int = 1000,
        low_w: int = 200,
        upper_w: int = 1000,
        max_ion_charge: int = 2,
        webscraping: str = "static",
        headless: bool = True,
        keep_browser_open: bool = False,
        detach_browser: bool = False,
    ):
        """

        Parameters
        ----------
        Te : float
             Electron temperature Te [eV]
        Ne: float
            Electron density Ne [cm^-3]
        elements: list[str]
            List of elements
        percentages: list[percentages]
            List of element percentages
        resolution: int
            Resoultion of spectrometer
        low_w: int
            Lower wavelength [nm]
        upper_w: int
            Upper wavelength [nm]
        max_ion_charge: int
            Maximal ion charge
        webscraping : str
            Type of webscraping: 'static' or 'dynamic'
        headless : bool
            Run Chrome in headless mode when using dynamic scraping
        keep_browser_open : bool
            Keep browser session open after dynamic scraping (useful for debugging)
        detach_browser : bool
            Detach Chrome from driver so browser can stay open after script exits

        """

        validate_simulated_libs(
            Te, Ne, elements, percentages, low_w, upper_w, max_ion_charge
        )

        self.Te = Te
        self.Ne = round(Ne, 3 - int(math.floor(math.log10(abs(Ne)))) - 1)
        self.Ne = re.sub(r"\+", "", str(self.Ne))

        self.elements = elements
        self.percentages = percentages
        self.resolution = resolution
        self.low_w = low_w
        self.upper_w = upper_w
        self.max_ion_charge = max_ion_charge

        self.raw_spectrum = pd.DataFrame({"wavelength": [], "intensity": []})
        self.interpolated_spectrum = pd.DataFrame({"wavelength": [], "intensity": []})
        self.webscraping = webscraping

        match webscraping:
            case "static":
                self.retrieve_data_static()
            case "dynamic":
                from selenium import webdriver
                from selenium.webdriver.common.by import By
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.webdriver.support.ui import WebDriverWait
                from webdriver_manager.chrome import ChromeDriverManager

                self.ion_spectra = None
                options = Options()
                options.add_argument("--disable-notifications")
                options.headless = headless
                if detach_browser:
                    options.add_experimental_option("detach", True)

                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
                self.driver.command_executor.client_config.timeout = (
                    CHROMEDRIVER_TRANSPORT_TIMEOUT_SECONDS
                )
                self.driver.set_page_load_timeout(BROWSER_TIMEOUT_SECONDS)
                try:
                    self.retrieve_data_dynamic()
                finally:
                    if not keep_browser_open:
                        self.driver.quit()
 

    def __repr__(self):
        return f"simLIBS(Te={self.Te:.2f} eV, Ne={self.Ne:.3e} cm^-3, elements={', '.join(self.elements)}, percentages={', '.join([str(p) for p in self.percentages])}, resolution={self.resolution}, low_w={self.low_w}, upper_w={self.upper_w}, max_ion_charge={self.max_ion_charge})"

    def __str__(self):
        return f"Te: {self.Te:.2f} eV, Ne: {self.Ne:.3e} cm^-3, elements: {', '.join(self.elements)}, percentages: {', '.join([str(p) for p in self.percentages])}"

    def get_site(self):
        """

        Returns
        -------

        """
        composition = ""
        spectrum = ""

        for i in range(len(self.elements)):
            if i > 0:
                composition += "3B"
                spectrum += "2C"
            composition += str(self.elements[i])
            composition += "%3A"
            composition += str(self.percentages[i])

            spectrum += str(self.elements[i])
            spectrum += "0-" + str(self.max_ion_charge)

            if i < len(self.elements) - 1:
                composition += "%"
                spectrum += "%"
        site = (
            "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?composition={}"
            "&spectra={}"
            "&low_w={}&limits_type=0&upp_w={}"
            "&show_av=2&unit=1"
            "&resolution={}"
            "&temp={}"
            "&eden={}"
            "&maxcharge={}"
            "&min_rel_int=0.001"
            "&int_scale=2"
            "&libs=1"
        )
        site = site.format(
            composition,
            spectrum,
            self.low_w,
            self.upper_w,
            self.resolution,
            self.Te,
            self.Ne,
            self.max_ion_charge,
        )

        return site

    def retrieve_data_dynamic(self):
        """

        Returns
        -------

        """
        site = self.get_site()
        self.driver.get(site)
        resolution_input = WebDriverWait(self.driver, BROWSER_TIMEOUT_SECONDS).until(
            EC.presence_of_element_located(
                (By.XPATH, "/html/body/div/div[1]/div[1]/form/div[3]/div/input")
            )
        )
        resolution_input.clear()
        resolution_input.send_keys(str(self.resolution))

        button_recalculate = WebDriverWait(self.driver, BROWSER_TIMEOUT_SECONDS).until(
            EC.presence_of_element_located(
                (By.XPATH, "/html/body/div/div[1]/div[1]/form/button")
            )
        )
        button_recalculate.click()

        button_csv = WebDriverWait(self.driver, BROWSER_TIMEOUT_SECONDS).until(
            EC.presence_of_element_located(
                (By.XPATH, "/html/body/div/div[2]/button[2]")
            )
        )
        button_csv.click()

        self.driver.switch_to.window((self.driver.window_handles[1]))

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        self.nist_csv_text = soup.pre.text.strip("\r\n")
        self.ion_spectra = pd.read_csv(io.StringIO(self.nist_csv_text), sep=",")
        self._set_spectrum_from_nist_csv()

    def retrieve_data_static(self):
        """Retrieve the same recalculated CSV as NIST without using Chrome."""
        session = self._new_static_session()
        response = session.get(
            self.get_site(), timeout=STATIC_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        page_scripts = [
            script.string or script.get_text()
            for script in soup.find_all("script")
            if "var dataDopplerArray" in str(script)
        ]
        if not page_scripts:
            raise ValueError(
                "NIST response did not contain LIBS spectrum data. "
                "Check the requested composition and wavelength range."
            )

        page_script = page_scripts[0]
        data_source = self._extract_nist_lte_data_source(page_script)
        recalculation_inputs = self._extract_recalculation_inputs(soup)
        lte_source = self._get_nist_lte_javascript(session)
        csv_text = self._run_nist_lte_javascript(
            data_source, lte_source, recalculation_inputs
        )
        self.nist_csv_text = csv_text.strip("\r\n")
        self.ion_spectra = pd.read_csv(io.StringIO(self.nist_csv_text), sep=",")
        self._set_spectrum_from_nist_csv()

    @staticmethod
    def _new_static_session():
        # The current project environment points requests at a flaky localhost
        # proxy. NIST is directly reachable, so static mode must not inherit it.
        session = requests.Session()
        session.trust_env = False
        return session

    @classmethod
    def _get_nist_lte_javascript(cls, session):
        if cls._nist_lte_javascript is not None:
            return cls._nist_lte_javascript

        with cls._nist_lte_javascript_lock:
            if cls._nist_lte_javascript is None:
                response = session.get(
                    NIST_LTE_JAVASCRIPT_URL,
                    timeout=STATIC_REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                if "function lte_spectrum_js" not in response.text:
                    raise ValueError(
                        "NIST Saha/LTE JavaScript response is incomplete."
                    )
                cls._nist_lte_javascript = response.text
        return cls._nist_lte_javascript

    @staticmethod
    def _extract_javascript_statement(source: str, variable: str) -> str:
        match = re.search(rf"\bvar\s+{re.escape(variable)}\s*=", source)
        if match is None:
            raise ValueError(f"NIST response is missing JavaScript variable {variable}.")

        quote = None
        escaped = False
        bracket_depth = 0
        brace_depth = 0
        parenthesis_depth = 0
        for index in range(match.end(), len(source)):
            char = source[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue

            if char in {"'", '"', "`"}:
                quote = char
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
            elif char == "(":
                parenthesis_depth += 1
            elif char == ")":
                parenthesis_depth -= 1
            elif (
                char == ";"
                and bracket_depth == 0
                and brace_depth == 0
                and parenthesis_depth == 0
            ):
                return source[match.start() : index + 1]

        raise ValueError(
            f"NIST JavaScript variable {variable} has no terminating semicolon."
        )

    @staticmethod
    def _extract_javascript_block(source: str, start_variable: str, end_variable: str):
        start = re.search(rf"\bvar\s+{re.escape(start_variable)}\s*=", source)
        end = re.search(rf"\bvar\s+{re.escape(end_variable)}\s*=", source)
        if start is None or end is None or start.start() >= end.start():
            raise ValueError(
                "NIST response has an unexpected LIBS data layout: "
                f"{start_variable}..{end_variable}."
            )
        return source[start.start() : end.start()]

    @classmethod
    def _extract_nist_lte_data_source(cls, page_script: str) -> str:
        statements = [
            cls._extract_javascript_statement(page_script, "composition"),
            cls._extract_javascript_statement(page_script, "koeff"),
            cls._extract_javascript_statement(page_script, "cmeV"),
            cls._extract_javascript_statement(page_script, "resolution"),
            cls._extract_javascript_statement(page_script, "temp"),
            cls._extract_javascript_statement(page_script, "eden"),
            cls._extract_javascript_statement(page_script, "IscaleType"),
            cls._extract_javascript_statement(page_script, "lines"),
            cls._extract_javascript_block(page_script, "spectra", "ips"),
            cls._extract_javascript_block(page_script, "ips", "gValues"),
            cls._extract_javascript_block(page_script, "gValues", "levels"),
            cls._extract_javascript_block(
                page_script, "levels", "dataDopplerArray"
            ),
            cls._extract_javascript_statement(page_script, "dataDopplerArray"),
        ]
        return "\n".join(statements)

    @staticmethod
    def _extract_recalculation_inputs(soup: BeautifulSoup) -> dict:
        composition_parts = []
        for percentage_input in soup.find_all("input", attrs={"name": "perc"}):
            input_id = percentage_input.get("id", "")
            element_label = soup.find(id="elem" + input_id.removeprefix("perc"))
            percentage = percentage_input.get("value")
            if element_label is None or percentage is None:
                raise ValueError(
                    "NIST response contains an incomplete composition form."
                )
            composition_parts.append(
                f"{element_label.get_text(strip=True)}:{percentage}"
            )

        def input_value(name):
            input_element = soup.find("input", attrs={"name": name})
            if input_element is None or input_element.get("value") is None:
                raise ValueError(f"NIST response is missing form value {name}.")
            return input_element["value"]

        if not composition_parts:
            raise ValueError("NIST response contains no composition form values.")
        return {
            "composition": ";".join(composition_parts),
            "temperature": input_value("temp"),
            "electron_density": input_value("eden"),
        }

    def _run_nist_lte_javascript(
        self,
        data_source: str,
        lte_source: str,
        recalculation_inputs: dict,
    ) -> str:
        node_executable = shutil.which("node")
        if node_executable is None:
            raise RuntimeError(
                "Static LIBS generation requires Node.js, but 'node' was not found "
                "on PATH. Install Node.js or use webscraping='dynamic'."
            )
        if not NIST_LTE_RUNNER.is_file():
            raise RuntimeError(f"Static LIBS runner is missing: {NIST_LTE_RUNNER}")

        payload = json.dumps(
            {
                "lteSource": lte_source,
                "dataSource": data_source,
                "resolution": self.resolution,
                "composition": recalculation_inputs["composition"],
                "temperature": recalculation_inputs["temperature"],
                "electronDensity": recalculation_inputs["electron_density"],
                "timeoutMilliseconds": STATIC_CALCULATION_TIMEOUT_SECONDS * 1000,
            }
        )
        try:
            completed = subprocess.run(
                [node_executable, str(NIST_LTE_RUNNER)],
                input=payload,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
                timeout=STATIC_CALCULATION_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                "Static NIST Saha/LTE calculation timed out after "
                f"{STATIC_CALCULATION_TIMEOUT_SECONDS} seconds."
            ) from error

        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown Node.js error"
            raise RuntimeError(detail)
        header = completed.stdout.partition("\n")[0]
        if not (
            header == "Wavelength (nm),Sum(calc)"
            or header.startswith("Wavelength (nm),Sum(calc),")
        ):
            raise ValueError("Static NIST calculation returned invalid CSV data.")
        return completed.stdout

    def _set_spectrum_from_nist_csv(self):
        required_columns = {"Wavelength (nm)", "Sum(calc)"}
        missing_columns = required_columns.difference(self.ion_spectra.columns)
        if missing_columns:
            raise ValueError(
                "NIST CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        spectrum = pd.DataFrame(
            {
                "wavelength": self.ion_spectra["Wavelength (nm)"],
                "intensity": self.ion_spectra["Sum(calc)"],
            }
        ).reset_index(drop=True)
        self.raw_spectrum = spectrum.copy()
        self.interpolated_spectrum = spectrum.copy()

    def retrieve_spectrum_from_html(self, html_data: str):

        start_index_of_spectrum_data = [
            (m.start(0), m.end(0))
            for m in re.finditer(r"var dataDopplerArray=.*", html_data)
        ]
        stop_index_of_spectrum_data = [
            (m.start(0), m.end(0))
            for m in re.finditer(r"]];\n    var dataSticksArray.*", html_data)
        ]
        spectrum_data = html_data[
            start_index_of_spectrum_data[0][1]
            + 1 : stop_index_of_spectrum_data[0][0]
            + 1
        ].split(",\n")
        i = 0
        for spectrums in spectrum_data:
            spectrum = spectrums[1:-1].split(",")
            self.raw_spectrum.loc[i] = [spectrum[0], spectrum[1]]
            i += 1

    def interpolate(self, resolution: float = 0.005):
        """
        interpolation of intensity with given resolution using CubicSpline
        """
        cs = CubicSpline(
            self.raw_spectrum["wavelength"],
            self.raw_spectrum["intensity"],
            bc_type="natural",
        )
        x = np.arange(self.low_w, self.upper_w, resolution)
        y = np.clip(cs(x), 0, np.inf)

        self.interpolated_spectrum["wavelength"] = np.round(x, 8)
        self.interpolated_spectrum["intensity"] = np.round(y, 8)

    def plot(
        self,
        color=(random.random(), random.random(), random.random()),
        title="Simulated LIBS",
    ):
        plt.plot(
            self.interpolated_spectrum["wavelength"],
            self.interpolated_spectrum["intensity"],
            label=str(self.elements) + str(self.percentages),
            color=color,
        )
        plt.grid()
        plt.title(title)
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel("Line Intensity [a.u.]")

    def plot_ion_spectra(self):
        self.ion_spectra.drop(["Sum(calc)"], axis=1).plot(
            x="Wavelength (nm)",
            xlabel=r"$\lambda$ [nm]",
            ylabel="Line Intensity [a.u.]",
            title="Ion spectra",
            grid=True,
        )

    def get_interpolated_spectrum(self):
        return self.interpolated_spectrum

    def get_raw_spectrum(self):
        return self.raw_spectrum

    def get_ion_spectra(self):

        if self.ion_spectra is not None:
            return self.ion_spectra
        else:
            raise ValueError(
                "NIST ion spectra are unavailable because data acquisition failed."
            )

    def save_to_csv(self, filepath: str, include_all_columns: bool = True):
        """Save the complete NIST CSV, or the legacy wavelength/intensity pair."""
        if include_all_columns:
            if not getattr(self, "nist_csv_text", None):
                raise ValueError("Complete NIST CSV data are unavailable.")
            with open(filepath, "w", encoding="utf-8", newline="") as csv_file:
                csv_file.write(self.nist_csv_text)
                csv_file.write("\n")
            return

        self.interpolated_spectrum.to_csv(path_or_buf=filepath, index=False)

    @staticmethod
    def worker(
        input_df: pd.DataFrame,
        Te_min: float,
        Te_max: float,
        Ne_min: float,
        Ne_max: float,
        webscraping: str,
    ):
        seed = random.randrange(len(input_df))
        percentages = input_df.iloc[seed].values[:-1]
        elements = input_df.iloc[seed].keys().values[:-1]
        name = input_df.iloc[seed]["name"]
        Te = random.uniform(Te_min, Te_max)
        Ne = random.uniform(Ne_min, Ne_max)
        fun = SimulatedLIBS(
            Te=Te,
            Ne=Ne,
            elements=elements,
            percentages=percentages,
            webscraping=webscraping,
        ).get_interpolated_spectrum()

        return {
            "spectrum": fun,
            "composition": pd.DataFrame(
                {"elements": elements, "percentages": percentages}
            ),
            "name": name,
            "Te[eV]": Te,
            "Ne[cm^-3]": Ne,
        }

    @staticmethod
    def create_dataset(
        input_composition_df: pd.DataFrame,
        size: int = 10,
        Te_min: float = 1.0,
        Te_max: float = 2.0,
        Ne_min: float = 10**17,
        Ne_max: float = 10**18,
        webscraping: str = "static",
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        input_composition_df: pd.DataFrame
            Input df with composition of elements to simulate
        size : int
            Output file size - number of simulated samples
        Te_min : float
            Minimal random electron temperature Te[eV]
        Te_max : float
            Maximal random electron temperature Te[eV]
        Ne_min : float
            Minimal random electron density Ne[cm^-3]
        Ne_max : float
            Maximal random electron density Ne[cm^-3]
        webscraping : str
            Webscraping type 'static' or 'dynamic'

        Returns
        -------

        """
        pool = ThreadPoolExecutor(size)
        spectra_pool = [
            pool.submit(
                SimulatedLIBS.worker,
                input_composition_df,
                Te_min,
                Te_max,
                Ne_min,
                Ne_max,
                webscraping,
            )
            for _ in range(size)
        ]
        columns = [
            str(wavelength)
            for wavelength in spectra_pool[0].result()["spectrum"]["wavelength"]
        ]
        for val in input_composition_df.columns.values:
            columns.append(str(val))
        columns.append("Te[eV]")
        columns.append("Ne[cm^-3]".format(Ne_min=Ne_min))
        output_df = pd.DataFrame(columns=columns)

        for spectra in spectra_pool:
            intensity = spectra.result()["spectrum"]["intensity"].values.tolist()
            percentages = spectra.result()["composition"]["percentages"].values.tolist()
            intensity.extend(percentages)
            intensity.append(spectra.result()["name"])
            intensity.append(spectra.result()["Te[eV]"])
            intensity.append(spectra.result()["Ne[cm^-3]".format(Ne_min=Ne_min)])
            output_df = pd.concat(
                [output_df, pd.DataFrame(data=[intensity], columns=columns)],
                ignore_index=True,
            )

        output_df.reset_index(drop=True)
        return output_df

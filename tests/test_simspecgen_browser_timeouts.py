from pathlib import Path
import re


SIMULATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "SimspecGen"
    / "simLIBS"
    / "simulation.py"
)


def test_dynamic_browser_waits_are_all_420_seconds():
    source = SIMULATION_PATH.read_text(encoding="utf-8")

    assert re.search(r"^BROWSER_TIMEOUT_SECONDS\s*=\s*420\s*$", source, re.MULTILINE)
    assert source.count(
        "self.driver.set_page_load_timeout(BROWSER_TIMEOUT_SECONDS)"
    ) == 1
    assert source.count("WebDriverWait(self.driver, BROWSER_TIMEOUT_SECONDS)") == 3

    numeric_waits = re.findall(r"WebDriverWait\(self\.driver,\s*\d+\)", source)
    assert numeric_waits == []


def test_chromedriver_transport_timeout_is_540_seconds():
    source = SIMULATION_PATH.read_text(encoding="utf-8")

    assert re.search(
        r"^CHROMEDRIVER_TRANSPORT_TIMEOUT_SECONDS\s*=\s*540\s*$",
        source,
        re.MULTILINE,
    )
    transport_timeout_assignments = re.findall(
        r"self\.driver\.command_executor\.client_config\.timeout\s*=\s*"
        r"\(?\s*CHROMEDRIVER_TRANSPORT_TIMEOUT_SECONDS\s*\)?",
        source,
    )
    assert len(transport_timeout_assignments) == 1

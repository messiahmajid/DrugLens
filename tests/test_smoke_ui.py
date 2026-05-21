"""
Smoke tests for DrugLens Screening Studio UI using Streamlit AppTest.

Verifies: app loads, sidebar controls exist, compound input tabs work,
sample CSV exists, screening button reachable, no raw Streamlit artifacts.

Run with:  python -m pytest tests/test_smoke_ui.py -v
"""

import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    return at


class TestAppLoads:
    def test_no_exception(self, app):
        assert not app.exception, f"App raised: {app.exception}"

    def test_title_present(self, app):
        all_text = " ".join(m.value for m in app.markdown)
        assert "DrugLens" in all_text


class TestSidebarControls:
    def test_target_group_radio_exists(self, app):
        radios = app.radio
        labels = [r.label for r in radios]
        assert "Target group" in labels

    def test_target_selectbox_exists(self, app):
        selects = app.selectbox
        labels = [s.label for s in selects]
        assert "Select target" in labels

    def test_filter_sliders_exist(self, app):
        slider_labels = [s.label for s in app.slider]
        assert "Min binding probability" in slider_labels
        assert "Max Lipinski violations" in slider_labels


class TestCompoundInput:
    def test_compound_source_radio(self, app):
        radios = app.radio
        labels = [r.label for r in radios]
        assert "Compound source" in labels

    def test_example_library_default(self, app):
        source_radio = [r for r in app.radio if r.label == "Compound source"][0]
        assert source_radio.value == "Example library"

    def test_multiselect_for_examples(self, app):
        ms = app.multiselect
        labels = [m.label for m in ms]
        assert "Select example compounds" in labels


class TestSampleCsv:
    def test_sample_csv_exists(self):
        assert Path("examples/sample_compounds.csv").exists()

    def test_sample_csv_has_header(self):
        text = Path("examples/sample_compounds.csv").read_text()
        assert "name,smiles" in text.lower()


class TestScreeningButton:
    def test_button_exists(self, app):
        buttons = app.button
        labels = [b.label for b in buttons]
        has_screen = any("Screen" in l or "Add compounds" in l for l in labels)
        assert has_screen, f"No screening button found. Labels: {labels}"


class TestNoRawArtifacts:
    def test_no_keyboard_arrow_right(self, app):
        all_text = " ".join(m.value for m in app.markdown)
        assert "keyboard_arrow_right" not in all_text

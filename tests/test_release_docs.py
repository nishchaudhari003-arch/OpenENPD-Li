"""
Smoke tests for release documentation and conservative scientific language.

These confirm the required docs exist, the README carries conservative status
phrases, the release docs avoid overclaim phrases (except where explicitly
negated), and the package exposes a version string.
"""

from pathlib import Path

import openenpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = (
    "README.md",
    "CHANGELOG.md",
    "docs/model_assumptions.md",
    "docs/validation_summary.md",
    "docs/limitations.md",
    "docs/release_checklist.md",
)

# Files authored/rewritten in the release-polish milestone, scanned for overclaim.
POLISH_FILES = REQUIRED_DOCS + ("docs/api_overview.md",)

# Positive overclaim phrases that must not appear (even negated versions of the
# claims are phrased so as not to contain these exact substrings).
FORBIDDEN_PHRASES = (
    "model validated",
    "physics fixed",
    "physics solved",
    "publication-ready",
    "publication ready",
    "manuscript-ready",
    "manuscript ready",
    "reproduces foo 2023",
    "accurate physical fit",
)


def test_required_docs_exist():
    for relative_path in REQUIRED_DOCS:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path


def test_readme_contains_conservative_status_phrases():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "diagnostic" in text
    assert "not final validation" in text
    assert "weakly identifiable" in text
    # The unresolved key result is stated honestly.
    assert "negative li" in text
    assert "remains unreproduced" in text or "does not" in text


def test_polish_files_avoid_overclaim_phrases():
    for relative_path in POLISH_FILES:
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in text, f"{phrase!r} found in {relative_path}"


def test_validation_summary_states_not_validation_success():
    text = (PROJECT_ROOT / "docs" / "validation_summary.md").read_text(
        encoding="utf-8"
    ).lower()
    assert "not validation success" in text


def test_repository_includes_mit_license():
    license_path = PROJECT_ROOT / "LICENSE"
    assert license_path.exists()
    license_text = license_path.read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Nishant Chaudhari" in license_text


def test_license_has_no_stray_trailing_text():
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    # The MIT text must end cleanly, with no stray trailing line.
    assert "Commit directly to main" not in license_text
    assert license_text.rstrip().endswith("DEALINGS IN THE\nSOFTWARE.")


def test_release_checklist_records_mit_license():
    text = (PROJECT_ROOT / "docs" / "release_checklist.md").read_text(
        encoding="utf-8"
    ).lower()
    assert "license" in text
    # The LICENSE now exists, so the checklist must record the MIT License rather
    # than flagging a pending decision.
    assert "mit license" in text


def test_package_exposes_version():
    assert isinstance(openenpd.__version__, str)
    assert openenpd.__version__ == "0.1.0"

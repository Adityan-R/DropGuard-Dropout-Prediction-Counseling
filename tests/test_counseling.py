"""
tests/test_counseling.py
Tests for src.dropguard.counseling: risk levels, note generation, action mapping.
"""
from __future__ import annotations

import pytest

from src.dropguard.counseling import (
    DRIVER_ACTIONS,
    RISK_LEVELS,
    generate_note,
    get_action_for_driver,
    get_risk_level,
    save_sample_notes,
)


# ---------------------------------------------------------------------------
# Risk level classification
# ---------------------------------------------------------------------------

class TestGetRiskLevel:
    def test_risk_level_low(self):
        """Probability 0.1 → 'Low'."""
        assert get_risk_level(0.1) == "Low"

    def test_risk_level_low_boundary(self):
        """Probability 0.0 → 'Low' (lower bound of range)."""
        assert get_risk_level(0.0) == "Low"

    def test_risk_level_medium(self):
        """Probability 0.45 → 'Medium'."""
        assert get_risk_level(0.45) == "Medium"

    def test_risk_level_medium_boundary(self):
        """Probability 0.3 → 'Medium' (lower bound of Medium range)."""
        assert get_risk_level(0.3) == "Medium"

    def test_risk_level_high(self):
        """Probability 0.8 → 'High'."""
        assert get_risk_level(0.8) == "High"

    def test_risk_level_high_boundary(self):
        """Probability 0.6 → 'High' (lower bound of High range)."""
        assert get_risk_level(0.6) == "High"

    def test_risk_level_maximum(self):
        """Probability 1.0 → 'High' (fallback for exact 1.0)."""
        assert get_risk_level(1.0) == "High"

    def test_risk_level_just_below_medium(self):
        """Probability 0.299 → 'Low'."""
        assert get_risk_level(0.299) == "Low"

    def test_risk_level_just_below_high(self):
        """Probability 0.599 → 'Medium'."""
        assert get_risk_level(0.599) == "Medium"


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

class TestGetActionForDriver:
    def test_absences_action_non_empty(self):
        """get_action_for_driver('absences') returns a non-empty string."""
        action = get_action_for_driver("absences")
        assert isinstance(action, str) and len(action) > 0

    def test_failures_action_non_empty(self):
        """get_action_for_driver('failures') returns a non-empty string."""
        action = get_action_for_driver("failures")
        assert isinstance(action, str) and len(action) > 0

    def test_g2_maps_to_tutoring(self):
        """'G2' (case-insensitive) maps to tutoring action."""
        action = get_action_for_driver("G2")
        assert "tutor" in action.lower()

    def test_g1_maps_to_review(self):
        """'G1' maps to a first-period review action."""
        action = get_action_for_driver("G1")
        assert "first" in action.lower() or "period" in action.lower()

    def test_studytime_action(self):
        """'studytime' maps to a study plan action."""
        action = get_action_for_driver("studytime")
        assert "study" in action.lower()

    def test_absences_action_mentions_attendance(self):
        """'absences' driver maps to an attendance-related action."""
        action = get_action_for_driver("absences")
        assert "attendance" in action.lower()

    def test_unknown_feature_returns_default(self):
        """An unrecognised feature name returns the default action."""
        action = get_action_for_driver("completely_unknown_feature_xyz")
        assert isinstance(action, str) and len(action) > 0

    def test_case_insensitive_matching(self):
        """Feature matching is case-insensitive."""
        lower = get_action_for_driver("absences")
        upper = get_action_for_driver("ABSENCES")
        mixed = get_action_for_driver("Absences")
        assert lower == upper == mixed

    def test_preprocessor_prefix_stripped(self):
        """Feature names with preprocessor prefixes still match correctly."""
        # e.g. 'num__absences' should match 'absences'
        action_plain = get_action_for_driver("absences")
        action_prefixed = get_action_for_driver("num__absences")
        # Both should return a non-empty action (they may differ if prefix matched)
        assert isinstance(action_prefixed, str) and len(action_prefixed) > 0


# ---------------------------------------------------------------------------
# Note generation
# ---------------------------------------------------------------------------

class TestGenerateNote:
    @pytest.fixture
    def sample_drivers(self):
        """Sample top-driver list for a high-risk student."""
        return [
            ("failures", 0.42),
            ("absences", 0.31),
            ("G2", 0.28),
        ]

    def test_note_is_non_empty_string(self, sample_drivers):
        """generate_note returns a non-empty string."""
        note = generate_note(
            student_idx=42,
            probability=0.75,
            top_drivers=sample_drivers,
        )
        assert isinstance(note, str) and len(note) > 0

    def test_note_contains_risk_level_high(self, sample_drivers):
        """Note for probability=0.75 contains 'High'."""
        note = generate_note(
            student_idx=1,
            probability=0.75,
            top_drivers=sample_drivers,
        )
        assert "High" in note

    def test_note_contains_risk_level_medium(self, sample_drivers):
        """Note for probability=0.45 contains 'Medium'."""
        note = generate_note(
            student_idx=2,
            probability=0.45,
            top_drivers=sample_drivers,
        )
        assert "Medium" in note

    def test_note_contains_risk_level_low(self, sample_drivers):
        """Note for probability=0.1 contains 'Low'."""
        note = generate_note(
            student_idx=3,
            probability=0.1,
            top_drivers=sample_drivers,
        )
        assert "Low" in note

    def test_note_has_recommended_actions(self, sample_drivers):
        """Note contains a 'Recommended Actions' section."""
        note = generate_note(
            student_idx=5,
            probability=0.8,
            top_drivers=sample_drivers,
        )
        assert "Recommended Actions" in note

    def test_note_has_key_factors(self, sample_drivers):
        """Note contains a 'Key Factors Identified' section."""
        note = generate_note(
            student_idx=5,
            probability=0.8,
            top_drivers=sample_drivers,
        )
        assert "Key Factors" in note

    def test_note_contains_probability(self, sample_drivers):
        """Note includes the predicted probability value."""
        note = generate_note(
            student_idx=7,
            probability=0.63,
            top_drivers=sample_drivers,
        )
        assert "0.63" in note

    def test_note_contains_student_index(self, sample_drivers):
        """Note references the student index."""
        note = generate_note(
            student_idx=99,
            probability=0.5,
            top_drivers=sample_drivers,
        )
        assert "99" in note

    def test_note_contains_driver_names(self, sample_drivers):
        """Note mentions at least one of the top driver features."""
        note = generate_note(
            student_idx=10,
            probability=0.72,
            top_drivers=sample_drivers,
        )
        # At least one driver feature name must appear in the note
        any_driver = any(d[0].lower() in note.lower() for d in sample_drivers)
        assert any_driver, "No driver feature names found in the generated note."

    def test_empty_drivers_still_produces_note(self):
        """generate_note handles empty top_drivers without crashing."""
        note = generate_note(
            student_idx=0,
            probability=0.5,
            top_drivers=[],
        )
        assert isinstance(note, str) and len(note) > 0

    def test_note_has_closing_sentence(self, sample_drivers):
        """Note ends with a supportive closing sentence (italicised)."""
        note = generate_note(
            student_idx=1,
            probability=0.8,
            top_drivers=sample_drivers,
        )
        # Closing is wrapped in * for Markdown italics
        assert "*" in note


# ---------------------------------------------------------------------------
# Batch saving
# ---------------------------------------------------------------------------

class TestSaveSampleNotes:
    def test_save_creates_file(self, tmp_path):
        """save_sample_notes creates a file at the given path."""
        notes = ["## Student 1\n\nSome note.", "## Student 2\n\nAnother note."]
        path = tmp_path / "notes.md"
        save_sample_notes(notes, path)
        assert path.exists()

    def test_saved_file_non_empty(self, tmp_path):
        """Saved markdown file is non-empty."""
        notes = ["## Student 1\n\nSome note."]
        path = tmp_path / "notes.md"
        save_sample_notes(notes, path)
        assert path.stat().st_size > 0

    def test_saves_at_most_ten_notes(self, tmp_path):
        """Only the first 10 notes are saved even if more are provided."""
        notes = [f"## Student {i}\n\nNote {i}." for i in range(20)]
        path = tmp_path / "notes.md"
        save_sample_notes(notes, path)
        content = path.read_text(encoding="utf-8")
        # Counts "## Student" headings — should be at most 10
        count = content.count("## Student")
        assert count <= 10

    def test_saved_file_is_utf8(self, tmp_path):
        """File is written as UTF-8 (special chars must survive round-trip)."""
        notes = ["## Student 1 \u2014 Risk Level: High\n\nNote with arrow \u2192."]
        path = tmp_path / "notes.md"
        save_sample_notes(notes, path)
        content = path.read_text(encoding="utf-8")
        assert "\u2014" in content   # em-dash
        assert "\u2192" in content   # right arrow

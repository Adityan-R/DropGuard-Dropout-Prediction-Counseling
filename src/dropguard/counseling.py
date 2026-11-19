"""
counseling.py - Rule/template-based counseling note generator for DropGuard.

No LLM or external API calls. All note generation is deterministic and based
on SHAP-identified risk drivers and configurable action templates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Risk-level thresholds
# ---------------------------------------------------------------------------

RISK_LEVELS: dict[tuple[float, float], str] = {
    (0.0, 0.3): "Low",
    (0.3, 0.6): "Medium",
    (0.6, 1.01): "High",
}

# ---------------------------------------------------------------------------
# Driver -> action mapping
# Keys are lowercase substrings matched against the feature name.
# Order matters: more specific keys should appear first.
# ---------------------------------------------------------------------------

DRIVER_ACTIONS: dict[str, str] = {
    "g2": "Schedule additional tutoring sessions and review recent assessments.",
    "g1": "Review first-period performance and identify early stumbling blocks.",
    "absences": "Conduct an attendance review meeting with the student and guardian.",
    "failures": "Assign a faculty mentor for weekly check-ins.",
    "studytime": "Create a structured weekly study plan with the student.",
    "medu": "Connect family with academic support resources.",
    "fedu": "Connect family with academic support resources.",
    "freetime": "Discuss healthy time-management strategies.",
    "goout": "Explore whether social commitments are affecting study time.",
    "health": "Refer student to campus health or wellness services.",
    "dalc": "Provide referral to student wellness counseling.",
    "walc": "Provide referral to student wellness counseling.",
    "famrel": "Engage school counselor for family support resources.",
    "internet": "Ensure student has access to school computer lab resources.",
    "schoolsup": "Review the extra academic support already offered by the school.",
    "course": "Provide subject-specific tutoring for this course.",
    "higher": (
        "Discuss the student's goals beyond secondary school and connect them "
        "with career and further-education guidance."
    ),
    "school": "Review the student's school placement and available on-site support.",
    "traveltime": "Arrange support with transport or scheduling given the commute.",
    "famsup": "Encourage the family to take part in the student's academic follow-up.",
    "paid": "Check whether additional paid tutoring is a viable option.",
    "age": "Confirm the student's placement fits their age and prior schooling.",
}

_DEFAULT_ACTION = "Schedule a general academic advising meeting."

# ---------------------------------------------------------------------------
# Supportive closing sentences per risk level
# ---------------------------------------------------------------------------

_CLOSINGS: dict[str, str] = {
    "Low": (
        "This student shows strong overall engagement; "
        "a brief check-in can help maintain their positive trajectory."
    ),
    "Medium": (
        "With timely support and clear guidance, "
        "this student has every opportunity to get back on track."
    ),
    "High": (
        "Immediate, coordinated support from academic advisors, faculty, and "
        "wellness staff is encouraged to help this student succeed."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_risk_level(probability: float) -> str:
    """Return ``'Low'``, ``'Medium'``, or ``'High'`` based on *probability*.

    Parameters
    ----------
    probability:
        Predicted at-risk probability in ``[0, 1]``.

    Returns
    -------
    str
        Risk category label.
    """
    for (low, high), label in RISK_LEVELS.items():
        if low <= probability < high:
            return label
    # Fallback for probability == 1.0
    return "High"


def get_action_for_driver(feature_name: str) -> str:
    """Return a recommended action string for the given *feature_name*.

    Matching is case-insensitive and substring-based, so ``'G2_scaled'``
    matches the ``'g2'`` key.

    Parameters
    ----------
    feature_name:
        Raw feature name string (may contain suffixes from the preprocessor).

    Returns
    -------
    str
        Recommended counseling action.
    """
    fn_lower = feature_name.lower()
    for pattern, action in DRIVER_ACTIONS.items():
        if pattern in fn_lower:
            return action
    return _DEFAULT_ACTION


def generate_note(
    student_idx: int,
    probability: float,
    top_drivers: list[tuple[str, float]],
    student_data: pd.Series | None = None,
) -> str:
    """Generate a short, supportive, non-judgmental counseling note.

    Parameters
    ----------
    student_idx:
        Original row index (used only for display).
    probability:
        Predicted at-risk probability in ``[0, 1]``.
    top_drivers:
        List of ``(feature_name, shap_value)`` tuples (highest impact first).
    student_data:
        Optional raw feature Series for the student (not currently used but
        available for future context enrichment).

    Returns
    -------
    str
        Formatted Markdown counseling note.

    Note Format
    -----------
    ::

        ## Student [idx] - Risk Level: [High/Medium/Low]
        **Predicted At-Risk Probability:** X.XX

        **Key Factors Identified:**
        - [Factor 1]: [direction] -> [Action]
        - [Factor 2]: [direction] -> [Action]
        - [Factor 3]: [direction] -> [Action]

        **Recommended Actions:**
        1. [Action 1]
        2. [Action 2]
        3. [Action 3]

        *[Supportive closing sentence based on risk level.]*
    """
    risk = get_risk_level(probability)

    lines: list[str] = [
        f"## Student {student_idx} \u2014 Risk Level: {risk}",
        f"**Predicted At-Risk Probability:** {probability:.2f}",
        "",
        "**Key Factors Identified:**",
    ]

    actions_seen: list[str] = []
    for feature_name, shap_val in top_drivers:
        direction = "elevated" if shap_val > 0 else "reduced"
        action = get_action_for_driver(feature_name)
        # Sanitise display name: strip preprocessor prefixes/suffixes
        display_name = feature_name
        for prefix in ("num__", "cat__", "remainder__"):
            if display_name.startswith(prefix):
                display_name = display_name[len(prefix):]
        display_name = (
            display_name.replace("_scaled", "")
            .replace("_encoded", "")
            .replace("_", " ")
            .strip()
        )
        lines.append(f"- **{display_name}** ({direction} influence) \u2192 {action}")
        if action not in actions_seen:
            actions_seen.append(action)

    lines.append("")
    lines.append("**Recommended Actions:**")
    for i, action in enumerate(actions_seen, start=1):
        lines.append(f"{i}. {action}")

    closing = _CLOSINGS.get(risk, _CLOSINGS["Medium"])
    lines.append("")
    lines.append(f"*{closing}*")
    lines.append("")

    return "\n".join(lines)


def generate_batch_notes(
    students_data: list[dict],
    probabilities: np.ndarray,
    X_test: pd.DataFrame,
) -> list[str]:
    """Generate counseling notes for a batch of students.

    Parameters
    ----------
    students_data:
        List of dicts as returned by
        :func:`~dropguard.explain.explain_students`.  Each dict must contain
        ``'student_idx'`` and ``'top_drivers'``.
    probabilities:
        1-D array of at-risk probabilities aligned with *students_data*.
    X_test:
        Raw feature DataFrame for the test set (used to extract per-student
        Series if needed).

    Returns
    -------
    list[str]
        One Markdown note string per student, in the same order as
        *students_data*.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    notes: list[str] = []

    for i, student_info in enumerate(students_data):
        student_idx = student_info["student_idx"]
        probability = float(probabilities[i])
        top_drivers: list[tuple[str, float]] = student_info["top_drivers"]

        # Retrieve raw Series if available
        student_data: pd.Series | None = None
        if student_idx in X_test.index:
            student_data = X_test.loc[student_idx]

        note = generate_note(
            student_idx=student_idx,
            probability=probability,
            top_drivers=top_drivers,
            student_data=student_data,
        )
        notes.append(note)

    return notes


def save_sample_notes(notes: list[str], save_path: Path) -> None:
    """Save the first 10 notes to a Markdown file.

    Parameters
    ----------
    notes:
        Full list of counseling note strings.
    save_path:
        Destination ``Path`` for the ``.md`` file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sample = notes[:10]
    header = (
        "# DropGuard - Sample Counseling Notes\n\n"
        f"> Auto-generated for {len(sample)} student(s). "
        "Provided for academic support staff review only.\n\n"
        "---\n\n"
    )

    content = header + "\n\n---\n\n".join(sample)

    save_path.write_text(content, encoding="utf-8")

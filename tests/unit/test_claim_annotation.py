"""Tests for dataset/claim_annotation.py."""
from __future__ import annotations

import pytest

from dataset.claim_annotation import (
    AnnotationRecord,
    adjudicate,
    cohens_kappa_path_correctness,
    cohens_kappa_verdicts,
    krippendorffs_alpha_path_correctness,
    krippendorffs_alpha_verdicts,
    load_annotation_csv,
    save_annotation_csv,
)


def _records(annotator_id, verdicts, path_correct=None):
    path_correct = path_correct or [None] * len(verdicts)
    return [
        AnnotationRecord(item_id=f"i{i}", dataset="demo", claim=f"c{i}", annotator_id=annotator_id,
                          verdict=v, path_correct=pc)
        for i, (v, pc) in enumerate(zip(verdicts, path_correct))
    ]


def test_perfect_agreement_kappa_is_one():
    a = _records("ann1", ["Supported", "Unsupported", "Contradictory"])
    b = _records("ann2", ["Supported", "Unsupported", "Contradictory"])
    assert cohens_kappa_verdicts(a, b) == pytest.approx(1.0)


def test_mismatched_item_sets_raises():
    a = _records("ann1", ["Supported"])
    b = [AnnotationRecord(item_id="different", dataset="demo", claim="c", annotator_id="ann2", verdict="Supported")]
    with pytest.raises(ValueError):
        cohens_kappa_verdicts(a, b)


def test_path_correctness_kappa_excludes_none_judgments():
    a = _records("ann1", ["Supported", "Supported", "Supported"], path_correct=[True, False, None])
    b = _records("ann2", ["Supported", "Supported", "Supported"], path_correct=[True, False, None])
    # i0/i1 have non-None judgments from both, with variation (True/False);
    # i2 is excluded since both left it unjudged.
    kappa = cohens_kappa_path_correctness(a, b)
    assert kappa == pytest.approx(1.0)


def test_krippendorff_alpha_handles_missing_items():
    a = _records("ann1", ["Supported", "Unsupported", "Contradictory"])
    b = _records("ann2", ["Supported", "Unsupported"])  # missing item i2
    alpha = krippendorffs_alpha_verdicts([a, b])
    assert -1.0 <= alpha <= 1.0


def test_krippendorff_alpha_requires_two_annotators():
    a = _records("ann1", ["Supported"])
    with pytest.raises(ValueError):
        krippendorffs_alpha_verdicts([a])


def test_krippendorff_alpha_path_correctness_excludes_none():
    a = _records("ann1", ["Supported", "Supported", "Supported"], path_correct=[True, False, None])
    b = _records("ann2", ["Supported", "Supported", "Supported"], path_correct=[True, False, None])
    alpha = krippendorffs_alpha_path_correctness([a, b])
    assert -1.0 <= alpha <= 1.0


def test_adjudicate_agreement_needs_no_adjudicator():
    records = _records("ann1", ["Supported"]) + [
        AnnotationRecord(item_id="i0", dataset="demo", claim="c0", annotator_id="ann2", verdict="Supported"),
    ]
    result = adjudicate("i0", "demo", "c0", records)
    assert result.disagreement is False
    assert result.final_verdict == "Supported"


def test_adjudicate_disagreement_requires_adjudicator_verdict():
    records = [
        AnnotationRecord(item_id="i0", dataset="demo", claim="c0", annotator_id="ann1", verdict="Supported"),
        AnnotationRecord(item_id="i0", dataset="demo", claim="c0", annotator_id="ann2", verdict="Unsupported"),
    ]
    with pytest.raises(ValueError):
        adjudicate("i0", "demo", "c0", records)

    result = adjudicate("i0", "demo", "c0", records, adjudicator_verdict="Contradictory", adjudicator_id="adj1")
    assert result.disagreement is True
    assert result.final_verdict == "Contradictory"


def test_adjudicate_path_correctness_majority_vote():
    records = [
        AnnotationRecord(item_id="i0", dataset="demo", claim="c0", annotator_id="ann1", verdict="Supported", path_correct=True),
        AnnotationRecord(item_id="i0", dataset="demo", claim="c0", annotator_id="ann2", verdict="Supported", path_correct=True),
        AnnotationRecord(item_id="i0", dataset="demo", claim="c0", annotator_id="ann3", verdict="Supported", path_correct=False),
    ]
    result = adjudicate("i0", "demo", "c0", records)
    assert result.final_path_correct is True


def test_csv_roundtrip(tmp_path):
    records = _records("ann1", ["Supported", "Contradictory"], path_correct=[True, None])
    path = str(tmp_path / "ann.csv")
    save_annotation_csv(records, path)
    loaded = load_annotation_csv(path)
    assert len(loaded) == 2
    assert loaded[0].verdict == "Supported"
    assert loaded[0].path_correct is True
    assert loaded[1].path_correct is None


def test_load_annotation_csv_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("item_id,claim\n1,c\n")
    with pytest.raises(ValueError):
        load_annotation_csv(str(path))

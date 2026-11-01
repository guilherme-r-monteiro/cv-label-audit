import random

from label_audit.audit import FindingKind, audit_image
from label_audit.corruption import Corruption, CorruptionKind, corrupt_image
from label_audit.detector import Detection
from label_audit.evaluate import ImageOutcome, random_queue_recall, score_threshold
from label_audit.labels import LabelBox

CAT_BOX = (100.0, 100.0, 200.0, 200.0)


def detection(label: str, box: tuple[float, float, float, float], confidence: float = 0.9) -> Detection:
    return Detection(label, confidence, box)


def test_matching_label_has_no_finding() -> None:
    assert audit_image([LabelBox("cat", CAT_BOX)], [detection("cat", CAT_BOX)], 640, 480) == []


def test_wrong_class_is_flagged_with_detector_label() -> None:
    findings = audit_image([LabelBox("dog", CAT_BOX)], [detection("cat", CAT_BOX)], 640, 480)
    assert [(f.kind, f.label) for f in findings] == [(FindingKind.WRONG_CLASS, "cat")]


def test_unlabeled_detection_is_missing_box() -> None:
    findings = audit_image([], [detection("cat", CAT_BOX)], 640, 480)
    assert [f.kind for f in findings] == [FindingKind.MISSING_BOX]


def test_label_without_detection_is_spurious() -> None:
    findings = audit_image([LabelBox("cat", CAT_BOX)], [], 640, 480)
    assert [f.kind for f in findings] == [FindingKind.SPURIOUS_BOX]


def test_partly_overlapping_same_class_is_bad_box() -> None:
    label = LabelBox("cat", (100.0, 100.0, 200.0, 200.0))
    findings = audit_image([label], [detection("cat", (170.0, 100.0, 270.0, 200.0))], 640, 480)
    assert [f.kind for f in findings] == [FindingKind.BAD_BOX]


def test_corruption_is_reproducible_for_a_seed() -> None:
    boxes = [LabelBox("cat", CAT_BOX)] * 40
    first = corrupt_image(boxes, random.Random(3), 640, 480)
    second = corrupt_image(boxes, random.Random(3), 640, 480)
    assert first == second
    assert {c.kind for c in first.corruptions} <= set(CorruptionKind)


def test_deleted_box_is_found_by_missing_box_finding() -> None:
    removed = LabelBox("cat", CAT_BOX)
    corruption = Corruption(CorruptionKind.DELETED, removed, None)
    findings = audit_image([], [detection("cat", CAT_BOX)], 640, 480)
    result = score_threshold([ImageOutcome(640, 480, [], [corruption], findings)], 0.25)
    assert (result.recall, result.precision) == (1.0, 1.0)


def test_low_score_findings_are_filtered_by_threshold() -> None:
    removed = LabelBox("cat", CAT_BOX)
    corruption = Corruption(CorruptionKind.DELETED, removed, None)
    findings = audit_image([], [detection("cat", CAT_BOX, 0.3)], 640, 480)
    assert score_threshold([ImageOutcome(640, 480, [], [corruption], findings)], 0.5).recall == 0.0


def test_random_queue_baseline_is_fraction_of_images() -> None:
    assert random_queue_recall(corrupted=50, total=100, size=10) == 0.5


def test_tiny_label_is_not_audited() -> None:
    tiny = LabelBox("cat", (10.0, 10.0, 20.0, 20.0))
    assert audit_image([tiny], [], 640, 480) == []

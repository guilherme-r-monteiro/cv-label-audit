from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from label_audit.detector import Detection, iou
from label_audit.labels import LabelBox

MATCH_IOU = 0.5
LOOSE_IOU = 0.1
UNLABELED_IOU = 0.3
AUDIT_CONFIDENCE = 0.25
SUPPORT_CONFIDENCE = 0.1
MINIMUM_AREA_FRACTION = 0.01


class FindingKind(StrEnum):
    MISSING_BOX = "missing_box"
    WRONG_CLASS = "wrong_class"
    BAD_BOX = "bad_box"
    SPURIOUS_BOX = "spurious_box"


@dataclass(frozen=True)
class Finding:
    kind: FindingKind
    label: str
    box: tuple[float, float, float, float]
    score: float
    label_index: int | None


def as_array(boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    return np.array(boxes, dtype=np.float64).reshape(-1, 4)


def overlaps(box: tuple[float, float, float, float], others: list[tuple[float, float, float, float]]) -> np.ndarray:
    if not others:
        return np.zeros(0)
    return iou(np.array(box, dtype=np.float64), as_array(others))


def area_fraction(box: tuple[float, float, float, float], width: int, height: int) -> float:
    return (box[2] - box[0]) * (box[3] - box[1]) / (width * height)


def judge_label(index: int, label: LabelBox, detections: list[Detection]) -> Finding | None:
    scores = overlaps(label.box, [d.box for d in detections])
    confident = [(scores[i], d) for i, d in enumerate(detections) if d.confidence >= AUDIT_CONFIDENCE]
    supported = any(d.label == label.label and scores[i] >= MATCH_IOU for i, d in enumerate(detections))
    if supported:
        return None
    other = [(s, d) for s, d in confident if d.label != label.label and s >= MATCH_IOU]
    if other:
        best = max(other, key=lambda pair: pair[1].confidence)[1]
        return Finding(FindingKind.WRONG_CLASS, best.label, label.box, best.confidence, index)
    loose = [(s, d) for s, d in confident if d.label == label.label and s >= LOOSE_IOU]
    if loose:
        best = max(loose, key=lambda pair: pair[1].confidence)[1]
        return Finding(FindingKind.BAD_BOX, label.label, label.box, best.confidence, index)
    return Finding(FindingKind.SPURIOUS_BOX, label.label, label.box, 1.0 - float(scores.max(initial=0.0)), index)


def explained_by_labels(detection: Detection, labels: list[LabelBox]) -> bool:
    scores = overlaps(detection.box, [label.box for label in labels])
    same_class = np.array([label.label == detection.label for label in labels], dtype=bool)
    return bool(((scores >= UNLABELED_IOU) | ((scores >= LOOSE_IOU) & same_class)).any())


def audit_image(labels: list[LabelBox], detections: list[Detection], width: int, height: int) -> list[Finding]:
    findings = [
        f
        for i, label in enumerate(labels)
        if area_fraction(label.box, width, height) >= MINIMUM_AREA_FRACTION
        if (f := judge_label(i, label, detections))
    ]
    findings += [
        Finding(FindingKind.MISSING_BOX, d.label, d.box, d.confidence, None)
        for d in detections
        if d.confidence >= AUDIT_CONFIDENCE
        and area_fraction(d.box, width, height) >= MINIMUM_AREA_FRACTION
        and not explained_by_labels(d, labels)
    ]
    return findings

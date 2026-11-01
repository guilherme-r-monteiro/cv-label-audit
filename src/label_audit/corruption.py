import random
from dataclasses import dataclass
from enum import StrEnum

from label_audit.labels import COCO_CLASSES, LabelBox

BOX_CORRUPTION_RATE = 0.03
SHIFT_FRACTION = 0.6
MINIMUM_SIDE = 4.0


class CorruptionKind(StrEnum):
    DELETED = "deleted"
    RELABELED = "relabeled"
    SHIFTED = "shifted"
    SPURIOUS = "spurious"


@dataclass(frozen=True)
class Corruption:
    kind: CorruptionKind
    original: LabelBox | None
    corrupted: LabelBox | None


@dataclass(frozen=True)
class CorruptedImage:
    boxes: list[LabelBox]
    corruptions: list[Corruption]


def shifted_box(box: LabelBox, rng: random.Random, width: int, height: int) -> LabelBox:
    x1, y1, x2, y2 = box.box
    dx = (x2 - x1) * SHIFT_FRACTION * rng.choice((-1, 1))
    dy = (y2 - y1) * SHIFT_FRACTION * rng.choice((-1, 1))
    moved = (max(0.0, x1 + dx), max(0.0, y1 + dy), min(float(width), x2 + dx), min(float(height), y2 + dy))
    return LabelBox(box.label, moved)


def relabeled_box(box: LabelBox, rng: random.Random) -> LabelBox:
    choices = [name for name in COCO_CLASSES if name != box.label]
    return LabelBox(rng.choice(choices), box.box)


def spurious_box(rng: random.Random, width: int, height: int) -> LabelBox:
    box_width = rng.uniform(0.1, 0.4) * width
    box_height = rng.uniform(0.1, 0.4) * height
    x1 = rng.uniform(0, width - box_width)
    y1 = rng.uniform(0, height - box_height)
    return LabelBox(rng.choice(COCO_CLASSES), (x1, y1, x1 + box_width, y1 + box_height))


def is_usable(box: LabelBox) -> bool:
    return box.box[2] - box.box[0] >= MINIMUM_SIDE and box.box[3] - box.box[1] >= MINIMUM_SIDE


def corrupt_image(boxes: list[LabelBox], rng: random.Random, width: int, height: int) -> CorruptedImage:
    kept: list[LabelBox] = []
    corruptions: list[Corruption] = []
    for box in boxes:
        if rng.random() >= BOX_CORRUPTION_RATE:
            kept.append(box)
            continue
        kind = rng.choice((CorruptionKind.DELETED, CorruptionKind.RELABELED, CorruptionKind.SHIFTED))
        if kind is CorruptionKind.DELETED:
            corruptions.append(Corruption(kind, box, None))
            continue
        changed = relabeled_box(box, rng) if kind is CorruptionKind.RELABELED else shifted_box(box, rng, width, height)
        if not is_usable(changed):
            kept.append(box)
            continue
        kept.append(changed)
        corruptions.append(Corruption(kind, box, changed))
    if boxes and rng.random() < BOX_CORRUPTION_RATE * len(boxes):
        extra = spurious_box(rng, width, height)
        kept.append(extra)
        corruptions.append(Corruption(CorruptionKind.SPURIOUS, None, extra))
    return CorruptedImage(kept, corruptions)

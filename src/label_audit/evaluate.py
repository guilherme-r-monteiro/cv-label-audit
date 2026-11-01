import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image

from label_audit.audit import (
    MINIMUM_AREA_FRACTION,
    SUPPORT_CONFIDENCE,
    Finding,
    FindingKind,
    area_fraction,
    audit_image,
    overlaps,
)
from label_audit.corruption import Corruption, CorruptionKind, corrupt_image
from label_audit.detector import Detection, OnnxDetector
from label_audit.labels import LabelBox, parse_label_file
from label_audit.store import open_store, review_queue, save_findings

SEEDS = (0, 1, 2, 3, 4)
SPLIT_INDEX = 64
SCORE_THRESHOLDS = (0.25, 0.5, 0.7)
QUEUE_SIZES = (10, 20)
DELETED_MATCH_IOU = 0.5
RELABEL_KINDS = frozenset({FindingKind.WRONG_CLASS})


@dataclass(frozen=True)
class SourceImage:
    name: str
    width: int
    height: int
    labels: list[LabelBox]
    detections: list[Detection]


@dataclass(frozen=True)
class ImageOutcome:
    width: int
    height: int
    boxes: list[LabelBox]
    corruptions: list[Corruption]
    findings: list[Finding]


@dataclass(frozen=True)
class ThresholdResult:
    min_score: float
    precision: float
    recall: float
    recall_all: float
    recall_by_kind: dict[str, float]
    findings: int
    corruptions: int


def load_sources(root: Path, detector: OnnxDetector) -> list[SourceImage]:
    images = sorted((root / "images" / "train2017").glob("*.jpg"))
    sources: list[SourceImage] = []
    for path in images:
        with Image.open(path) as handle:
            width, height = handle.size
        labels = parse_label_file(root / "labels" / "train2017" / f"{path.stem}.txt", width, height)
        sources.append(SourceImage(path.name, width, height, labels, detector.detect(path, SUPPORT_CONFIDENCE)))
    return sources


def finding_hits(corruption: Corruption, boxes: list[LabelBox], findings: list[Finding]) -> list[Finding]:
    if corruption.kind is CorruptionKind.DELETED and corruption.original:
        return [
            f
            for f in findings
            if f.kind is FindingKind.MISSING_BOX
            and overlaps(f.box, [corruption.original.box])[0] >= DELETED_MATCH_IOU
        ]
    index = boxes.index(corruption.corrupted) if corruption.corrupted in boxes else -1
    on_label = [f for f in findings if f.label_index == index]
    if corruption.kind is CorruptionKind.RELABELED:
        return [f for f in on_label if f.kind in RELABEL_KINDS]
    return on_label


def is_auditable(corruption: Corruption, width: int, height: int) -> bool:
    box = corruption.original or corruption.corrupted
    return box is not None and area_fraction(box.box, width, height) >= MINIMUM_AREA_FRACTION


def score_threshold(outcomes: list[ImageOutcome], min_score: float) -> ThresholdResult:
    found: dict[str, int] = {}
    total: dict[str, int] = {}
    all_total = all_found = flagged = explained = 0
    for outcome in outcomes:
        kept = [f for f in outcome.findings if f.score >= min_score]
        flagged += len(kept)
        hit_ids: set[int] = set()
        for corruption in outcome.corruptions:
            hits = finding_hits(corruption, outcome.boxes, kept)
            all_total += 1
            all_found += bool(hits)
            if not is_auditable(corruption, outcome.width, outcome.height):
                continue
            total[corruption.kind.value] = total.get(corruption.kind.value, 0) + 1
            found[corruption.kind.value] = found.get(corruption.kind.value, 0) + bool(hits)
            hit_ids.update(id(h) for h in hits)
        explained += len(hit_ids)
    auditable = sum(total.values())
    return ThresholdResult(
        min_score,
        explained / flagged if flagged else 0.0,
        sum(found.values()) / auditable if auditable else 0.0,
        all_found / all_total if all_total else 0.0,
        {kind: found.get(kind, 0) / count for kind, count in total.items()},
        flagged,
        auditable,
    )


def queue_recall(
    ranked: list[tuple[str, int, float]], corrupted_images: set[str], size: int
) -> float:
    top = {name for name, _, _ in ranked[:size]}
    return len(top & corrupted_images) / min(size, len(corrupted_images)) if corrupted_images else 0.0


def random_queue_recall(corrupted: int, total: int, size: int) -> float:
    if not corrupted:
        return 0.0
    return size * corrupted / total / min(size, corrupted)


def run_seed(sources: list[SourceImage], seed: int, database: Path | None) -> dict[str, object]:
    rng = random.Random(seed)
    outcomes: list[ImageOutcome] = []
    corrupted_images: set[str] = set()
    connection = open_store(database) if database else open_store(Path(":memory:"))
    for source in sources:
        corrupted = corrupt_image(source.labels, rng, source.width, source.height)
        findings = audit_image(corrupted.boxes, source.detections, source.width, source.height)
        outcomes.append(ImageOutcome(source.width, source.height, corrupted.boxes, corrupted.corruptions, findings))
        save_findings(connection, source.name, findings)
        if corrupted.corruptions:
            corrupted_images.add(source.name)
    ranked = review_queue(connection, len(sources))
    queue = {
        str(size): {
            "audit": queue_recall(ranked, corrupted_images, size),
            "random_expected": random_queue_recall(len(corrupted_images), len(sources), size),
        }
        for size in QUEUE_SIZES
    }
    return {
        "seed": seed,
        "corrupted_images": len(corrupted_images),
        "thresholds": [asdict(score_threshold(outcomes, t)) for t in SCORE_THRESHOLDS],
        "review_queue_recall": queue,
    }


def clean_noise_floor(sources: list[SourceImage]) -> dict[str, float]:
    findings = [audit_image(s.labels, s.detections, s.width, s.height) for s in sources]
    flagged_images = sum(1 for f in findings if f)
    return {
        "images": len(sources),
        "labels": sum(len(s.labels) for s in sources),
        "findings": sum(len(f) for f in findings),
        "images_with_findings": flagged_images,
    }


def main(dataset_root: Path, model_path: Path, output: Path, database: Path | None) -> None:
    sources = load_sources(dataset_root, OnnxDetector(model_path))
    report = {
        "clean_noise_floor": clean_noise_floor(sources),
        "seeds": [run_seed(sources, seed, database if seed == SEEDS[0] else None) for seed in SEEDS],
    }
    output.write_text(json.dumps(report, indent=2))
    sys.stdout.write(json.dumps(report["clean_noise_floor"]) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]) if len(sys.argv) > 4 else None)

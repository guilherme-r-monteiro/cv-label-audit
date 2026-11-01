import json
import statistics
import sys
from pathlib import Path

KINDS = ("deleted", "relabeled", "shifted", "spurious")


def mean_of(seeds: list[dict], index: int, key: str) -> float:
    return statistics.mean(seed["thresholds"][index][key] for seed in seeds)


def kind_recall(seeds: list[dict], index: int, kind: str) -> float:
    return statistics.mean(seed["thresholds"][index]["recall_by_kind"].get(kind, 0.0) for seed in seeds)


def threshold_table(seeds: list[dict]) -> list[str]:
    header = "| Min score | Precision | Recall (auditable) | Recall (all) | " + " | ".join(KINDS) + " |"
    lines = [header, "|" + "---|" * (5 + len(KINDS))]
    for index, threshold in enumerate(seeds[0]["thresholds"]):
        cells = [f"{threshold['min_score']}", f"{mean_of(seeds, index, 'precision'):.2f}"]
        cells += [f"{mean_of(seeds, index, 'recall'):.2f}", f"{mean_of(seeds, index, 'recall_all'):.2f}"]
        cells += [f"{kind_recall(seeds, index, kind):.2f}" for kind in KINDS]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def queue_table(seeds: list[dict]) -> list[str]:
    lines = ["| Queue size | Audit | Random |", "|---|---|---|"]
    for size in seeds[0]["review_queue_recall"]:
        audit = statistics.mean(s["review_queue_recall"][size]["audit"] for s in seeds)
        random_ = statistics.mean(s["review_queue_recall"][size]["random_expected"] for s in seeds)
        lines.append(f"| {size} | {audit:.2f} | {random_:.2f} |")
    return lines


def main(path: Path) -> None:
    report = json.loads(path.read_text())
    seeds = report["seeds"]
    sys.stdout.write("\n".join(threshold_table(seeds) + [""] + queue_table(seeds)) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

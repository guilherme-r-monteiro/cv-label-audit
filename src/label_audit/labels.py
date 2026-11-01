from dataclasses import dataclass
from pathlib import Path

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class LabelBox:
    label: str
    box: tuple[float, float, float, float]


def parse_label_file(label_file: Path, width: int, height: int) -> list[LabelBox]:
    if not label_file.exists():
        return []
    boxes: list[LabelBox] = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        center_x, center_y, box_width, box_height = (float(v) for v in parts[1:])
        boxes.append(
            LabelBox(
                COCO_CLASSES[int(parts[0])],
                (
                    (center_x - box_width / 2) * width,
                    (center_y - box_height / 2) * height,
                    (center_x + box_width / 2) * width,
                    (center_y + box_height / 2) * height,
                ),
            )
        )
    return boxes

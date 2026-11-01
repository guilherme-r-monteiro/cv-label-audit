# cv-label-audit

Finds label errors in an object-detection dataset with a detector, stores the findings in SQLite, ranks images for review with SQL, and measures the audit against errors injected with a fixed seed.

- Data: COCO128 (128 images, 929 boxes).
- Detector: YOLOv8n exported to ONNX, run on the CPU with ONNX Runtime.
- Findings: `missing_box`, `wrong_class`, `bad_box`, `spurious_box`.
- Store: SQLite table `findings`. The review queue is one SQL query (`store.py`).
- Design note: `docs/design.md`.

## Results

Five seeds, each injecting about 40 auditable errors at a 3% per-box rate. Numbers are the mean over the seeds. Raw rows are in `results/audit-results.json`.

| Min score | Precision | Recall (auditable) | Recall (all) | deleted | relabeled | shifted | spurious |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.23 | 0.89 | 0.69 | 0.30 | 0.68 | 0.96 | 1.00 |
| 0.5 | 0.27 | 0.85 | 0.66 | 0.26 | 0.52 | 0.84 | 1.00 |
| 0.7 | 0.30 | 0.69 | 0.53 | 0.15 | 0.38 | 0.84 | 0.80 |

Review queue, share of the top k images that contain an injected error:

| Queue size | Audit | Random |
|---|---|---|
| 10 | 0.72 | 0.31 |
| 20 | 0.68 | 0.31 |

On the unmodified dataset the audit raises 131 findings on 929 labels, in 61 of 128 images. This is the noise floor: real label errors plus detector errors.

What the numbers say:
- The audit finds most wrong-position and invented boxes, and about two thirds of wrong classes at the lowest threshold.
- It finds few deleted boxes (0.30). A deleted object is found only when the detector sees it with confidence.
- Precision is low (0.23 to 0.30). Most flags are detector disagreements, not injected errors. The queue helps a reviewer order work. It does not replace review.

## Limits

- 128 images and five seeds. Differences of a few points between thresholds are noise.
- YOLOv8n was trained on COCO, which includes these images. The detector agrees with the labels more than it would on new data, so recall is optimistic.
- Injected errors are synthetic. Real annotation errors differ in kind and frequency.
- Boxes under 1% of the image are not audited. Recall for all errors is lower for that reason.
- The audit was revised once after the first run. The reason and the first-run numbers are in `docs/design.md`. The final numbers come from a design that was seen on this data.

## Run it

```sh
scripts/setup.sh
PYTHONPATH=src .venv/bin/python -m pytest
PYTHONPATH=src .venv/bin/python -m label_audit.evaluate data/coco128 models/yolov8n.onnx results/audit-results.json results/findings-seed0.db
PYTHONPATH=src .venv/bin/python -m label_audit.report results/audit-results.json
```

## Layout

- `src/label_audit/detector.py`: letterbox, ONNX inference, decoding, non-max suppression.
- `src/label_audit/corruption.py`: seeded error injection.
- `src/label_audit/audit.py`: label and detection comparison.
- `src/label_audit/store.py`: SQLite schema and queries.
- `src/label_audit/evaluate.py`, `report.py`: scoring and tables.
- `tests/`: 10 unit tests.

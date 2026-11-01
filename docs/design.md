# Design note

Target: BairesDev Senior Machine Learning Engineer (Computer Vision), and the BairesDev Computer Vision Engineer (AV) posting.
Those postings ask for model work on detection and segmentation, data pipelines, data mining, training data quality, SQL and model evaluation.

## Question

Can a detector find label errors in an object-detection dataset well enough to build a review queue, and how much noise does it add?

## Method

1. Run YOLOv8n (ONNX) once per image at confidence 0.1 and keep every detection.
2. Inject label errors with a fixed seed: deleted box, wrong class, shifted box, spurious box. Rate: 3% per box.
3. Compare labels and detections. Raise one finding per problem: `missing_box`, `wrong_class`, `bad_box`, `spurious_box`.
4. Store findings in SQLite. Rank images with SQL by the sum of finding scores. This is the review queue.
5. Score against the injected errors, five seeds, three score thresholds.

## Scope limits chosen before the run

- Boxes smaller than 1% of the image area are not audited. The detector cannot judge them.
- Recall is reported for auditable errors and for all errors.
- Precision counts a finding as correct only when it matches an injected error. Real label errors and detector errors both count as wrong, so precision is a lower bound.

## Change after the first run

The first version audited all boxes and treated every label without a detection at confidence 0.25 as spurious.
On the unmodified dataset it raised 534 findings on 929 labels, and 394 were spurious-box findings, mostly small objects.
That made the review queue useless, so the small-box floor and the 0.1 support confidence were added.

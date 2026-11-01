#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
uv venv .venv
uv pip install --python .venv/bin/python -e . pytest ultralytics
mkdir -p data models
curl -sL -o data/coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
unzip -q -o data/coco128.zip -d data
.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=12)"
mv yolov8n.onnx models/yolov8n.onnx
rm -f yolov8n.pt

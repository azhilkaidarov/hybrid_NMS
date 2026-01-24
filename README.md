# Bounding Box Project

A Python project for bounding box operations: IoU calculation, hybrid NMS, and visualization.

## Features

- **IoU Calculation**: Compute Intersection over Union for bounding boxes
- **Hybrid NMS**: Two-stage Non-Maximum Suppression (merge + delete)
- **Visualization**: Draw bounding boxes with labels and confidence scores
- **Synthetic Data**: Generate random bounding boxes for testing

## Project Structure

```
src/
├── bounding_box_project.py   # Main entry point
├── boxes.py                  # Bbox dataclass
├── iou.py                    # IoU calculation
├── nms.py                    # Hybrid NMS algorithm
├── viz.py                    # Visualization utilities
├── image_io.py               # Image load/save helpers
└── synthetic.py              # Random bbox generation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/azhilkaidarov/hybrid_NMS.git
cd hybrid_NMS

# Install dependencies
uv sync
```

## Usage

### Command Line

```bash
# Run demo with GUI window
uv run python -m bounding_box_project --show

# Custom parameters
uv run python -m bounding_box_project \
    --image data/input/sample_image.png \
    --n 10 \
    --threshold 0.2 \
    --merge-threshold 0.6 \
    --out-dir data/outputs
```

### As a Library

```python
from boxes import Bbox
from iou import iou
from nms import hybrid_nms

# Create bounding boxes
b1 = Bbox(x_min=0, y_min=0, x_max=10, y_max=10, confidence=0.9)
b2 = Bbox(x_min=5, y_min=5, x_max=15, y_max=15, confidence=0.7)

# Calculate IoU
score = iou(b1, b2)  # 0.143

# Apply hybrid NMS
bboxes = [b1, b2]
result = hybrid_nms(bboxes, threshold=0.2, merge_iou_threshold=0.6)
```

## Dependencies

- numpy
- opencv-python

## License

MIT

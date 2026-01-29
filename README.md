# Vision-Language-Audio Perception System for Assistive Navigation

A multi-modal AI system designed to assist visually impaired individuals with real-time navigation feedback. The system integrates object detection, scene captioning, optical character recognition, and text-to-speech synthesis.

## Overview

This project implements an end-to-end assistive vision pipeline that:
1. Detects objects relevant to navigation safety (vehicles, pedestrians, obstacles, traffic signs)
2. Generates natural language descriptions of scenes
3. Reads text from signs and labels
4. Converts all information to audio feedback

## Architecture

```
INPUT IMAGE
      │
      ▼
┌─────────────────┐
│ YOLOv8 Detection│──► Objects + Text Regions
└─────────────────┘
      │
      ├──────────────────────┐
      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐
│ BLIP-2 Caption  │   │   TrOCR OCR     │
└─────────────────┘   └─────────────────┘
      │                      │
      └──────────┬───────────┘
                 ▼
        ┌─────────────────┐
        │  Fusion Layer   │
        └─────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  gTTS Audio     │
        └─────────────────┘
                 │
                 ▼
           AUDIO OUTPUT
```

## Features

- **Object Detection**: YOLOv8s fine-tuned on 28 classes (COCO + Road Signs)
- **Scene Captioning**: BLIP-2 with hallucination reduction
- **Text Recognition**: TrOCR for reading signs and labels
- **Audio Output**: Real-time text-to-speech feedback
- **Priority Classes**: Safety-critical objects (vehicles, pedestrians, traffic signs) are prioritized

## Installation

### Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install transformers accelerate
pip install gtts pyttsx3
```

### Dependencies

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- Ultralytics (YOLOv8)
- Transformers (BLIP-2, TrOCR)
- gTTS (Text-to-Speech)

## Usage

### Running on Kaggle (Recommended)

1. Upload the notebook to Kaggle
2. Add the following datasets:
   - `coco-2014-dataset-for-yolov3` or `coco-image-caption`
   - `road-sign-detection`
3. Enable GPU accelerator (P100 or T4)
4. Run all cells

### Quick Start

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("yolo_assistive_vision.pt")

# Run inference
results = model("path/to/image.jpg")
```

### Full Pipeline

```python
# Initialize the system
system = AssistiveVisionSystem("yolo_assistive_vision.pt")

# Analyze an image
result = system.analyze("path/to/image.jpg", visualize=True)

# Access results
print(result['caption'])           # Scene description
print(result['detections'])        # Detected objects
print(result['audio_description']) # Audio feedback text

# Play audio
audio_system.play_feedback(result['audio_description'])
```

## Dataset

The model is trained on a combined dataset:

| Dataset | Images | Classes |
|---------|--------|---------|
| COCO 2014 (filtered) | ~10,000 | 24 assistive-relevant classes |
| Road Signs | ~877 | 4 classes (traffic light, stop, speed limit, crosswalk) |

### Class Categories

**Priority Classes (Safety-Critical)**
- person, car, bicycle, motorcycle, bus, truck
- traffic light, stop sign, crosswalk

**Navigation Obstacles**
- bench, chair, couch, potted plant, dining table

**Indoor Objects**
- tv, laptop, cell phone, book, clock, bottle, cup, etc.

## Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | See training results |
| mAP@50-95 | See training results |
| Inference Speed | ~30-50 FPS (GPU) |

## Project Structure

```
├── vision-language-audio-perception-system-for-assist.ipynb  # Main notebook
├── README.md                                                  # This file
└── final_model/                                               # Generated outputs
    ├── yolo_assistive_vision.pt                              # Trained model
    ├── class_config.json                                      # Class mapping
    ├── system_config.json                                     # Configuration
    └── training_results/                                      # Training curves
```

## Training

To retrain the model:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="data.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device=0
)
```

### Training Configuration

- **Base Model**: YOLOv8s (pretrained on COCO)
- **Epochs**: 100
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Augmentation**: Mosaic, MixUp, geometric transforms

## Output Format

The system returns a structured result:

```python
{
    'detections': [
        {'class': 'person', 'confidence': 0.95, 'bbox': [...], 'is_priority': True},
        ...
    ],
    'caption': "A person walking on a sidewalk near a parked car",
    'ocr_texts': [{'text': 'STOP', 'source': 'stop sign'}],
    'audio_description': "Attention: person nearby on left. A person walking..."
}
```

## Limitations

- Requires GPU for real-time performance
- OCR accuracy depends on image quality and text visibility
- Caption quality varies with scene complexity

## Future Improvements

- [ ] Add depth estimation for distance measurement
- [ ] Implement real-time video processing
- [ ] Add support for multiple languages
- [ ] Optimize for edge devices (mobile, Raspberry Pi)

## References

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [TrOCR](https://huggingface.co/microsoft/trocr-base-printed)
- [COCO Dataset](https://cocodataset.org/)

## Author

**Aayush Khanal**  
Student ID: 20049123

## License

This project is for educational purposes as part of a Deep Learning coursework.

Release link:
https://github.com/HALF-BLOOD/Vision-Language-Audio-Perception-System-/releases/tag/VisionLanguageAudioPerception

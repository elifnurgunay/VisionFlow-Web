# 🚀 VisionFlow Web Studio

VisionFlow Web Studio is a modern web-based application that provides video frame extraction, automatic labeling, and data augmentation operations within a single, intuitive interface. Ported from the original desktop version to a Flask-based web environment, it offers a seamless workflow for YOLO dataset preparation.

![VisionFlow Web Interface](https://raw.githubusercontent.com/elifnurgunay/VisionFlow-Web/main/static/preview1.png)
![VisionFlow Analysis](https://raw.githubusercontent.com/elifnurgunay/VisionFlow-Web/main/static/preview2.png)

## 1. FEATURES

### 1. Frame Extractor
Extracts frames from video files for model training.
- **Interval Sampling:** Saves every Nth frame.
- **Time-Based Extraction:** Saves a specified number of frames per second (Target FPS).
- **Motion Threshold:** Saves only distinct scenes and skips nearly identical frames using difference scoring.
- **Preview:** Quickly previews the video using 6 random frames.
- **Video Information:** Automatically displays duration, FPS, resolution, and total frame count.
- **Automatic Packaging:** Automatically generates a `.zip` archive of all extracted frames for easy download.

### 2. Auto Label (Upcoming)
Automatically labels images using a trained YOLO model.
- Generates annotation files in `.txt` and `.json` formats.
- Supports GPU acceleration with CUDA.
- Provides statistical reports and visualization.

### 3. Augmentation (Upcoming)
Artificially expands the dataset using the **Albumentations** framework.
- Supports Geometric, Photometric, Blur, and Sensor Noise transformations.
- Fully configurable parameters through the web UI.

## 2. USAGE

### Requirements
- Python 3.10+
- CUDA 12.x (Optional, for GPU acceleration in Auto Label)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/elifnurgunay/VisionFlow-Web.git
cd VisionFlow-Web

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000` to start.

## 3. PROJECT STRUCTURE

```text
VisionFlow-Web/
│
├── app.py                  # Entry point — Flask Server, API endpoints
├── core/
│   ├── extractor.py        # Frame extraction logic
│   ├── labeler.py          # Auto-label pipeline
│   └── augmentor.py        # Augmentation pipeline
│
├── templates/
│   └── index.html          # Main web interface (Bootstrap 5)
│
├── uploads/                # Temporary storage for uploaded videos
├── outputs/                # Extracted frames and ZIP archives
│
├── config.yaml             # Model and directory configuration
└── requirements.txt        # Project dependencies
```

## 4. DEPENDENCIES

| Package | Purpose |
| :--- | :--- |
| **Flask** | Web Framework |
| **OpenCV** | Image Processing and video handling |
| **Ultralytics** | YOLO model inference |
| **Albumentations** | Data Augmentation |
| **PyTorch** | Deep Learning operations |
| **PyYAML** | Configuration management |

## 5. CREDITS / TEAM

This project is developed by:
- **Zehra Kelahmetoğlu**
- **Atakan Yılmaz**
- **Zeynep Ötegen**
- **Elif Nur Günay**
- **Sevda Tuba Ehlibeyt**

## 6. LICENSE

This project is licensed under the **MIT License**.

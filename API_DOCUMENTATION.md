# VisionFlow Web Studio API Documentation

This document outlines the RESTful API endpoints for **VisionFlow Web Studio**, a comprehensive tool for YOLO dataset preparation, including frame extraction, automated labeling, and data augmentation.

## Base URL
All API requests should be directed to the following base URL:
`http://localhost:5000`

---

## 1. Video Analysis
This resource manages the initial inspection and metadata retrieval of video files.

### Get Video Metadata and Previews
This endpoint uploads a video file to retrieve its technical specifications (FPS, duration, resolution) and generates six random preview frames.

- **Endpoint:** `/get_video_data`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

#### Parameters
| Name | Type | Location | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `video` | File | Body | **Yes** | The video file to analyze (.mp4, .avi, .mkv). |

#### Request Example
```bash
curl -X POST http://localhost:5000/get_video_data \
  -F "video=@/path/to/your/video.mp4"
```

#### Response JSON Schema
```json
{
  "status": "success",
  "info": {
    "duration": 15.5,
    "fps": 30.0,
    "height": 1080,
    "total_frames": 465,
    "width": 1920
  },
  "previews": [
    "iVBORw0KGgoAAAANSUhEUg...",
    "..."
  ]
}
```

---

## 2. Frame Extraction
This resource handles the conversion of video files into individual image frames for training datasets.

### Extract Frames from Video
This endpoint processes an uploaded video and extracts frames based on the selected sampling mode.

- **Endpoint:** `/extract`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

#### Parameters
| Name | Type | Location | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `video` | File | Body | **Yes** | The source video file. |
| `mode` | String | Body | **Yes** | Extraction strategy: `nth`, `fps`, or `motion`. |
| `interval` | Integer | Body | Optional | Sampling interval for `nth` mode (default: 10). |
| `target_fps` | Float | Body | Optional | Target frame rate for `fps` mode (default: 2.0). |
| `motion_thresh` | Float | Body | Optional | Difference threshold for `motion` mode (default: 15.0). |

#### Request Example
```bash
curl -X POST http://localhost:5000/extract \
  -F "video=@/path/to/video.mp4" \
  -F "mode=nth" \
  -F "interval=20"
```

#### Response JSON Schema
```json
{
  "status": "success",
  "count": 23,
  "download_url": "/download/dataset_frames.zip"
}
```

---

## 3. Automated Labeling
This resource utilizes YOLO models to automatically generate annotation files for the extracted frames.

### Run Auto Labeling Pipeline
This endpoint runs a pre-trained YOLO model on the previously extracted frames to generate bounding box labels.

- **Endpoint:** `/autolabel`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

#### Parameters
| Name | Type | Location | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `conf_thresh` | Float | Body | Optional | Confidence threshold for detection (default: 0.5). |
| `iou_thresh` | Float | Body | Optional | Intersection over Union threshold (default: 0.45). |
| `img_size` | Integer | Body | Optional | Input image size for the model (default: 640). |

#### Request Example
```bash
curl -X POST http://localhost:5000/autolabel \
  -F "conf_thresh=0.6" \
  -F "img_size=640"
```

#### Response JSON Schema
```json
{
  "status": "success",
  "count": 150,
  "download_url": "/download/dataset_labeled_results.zip"
}
```

---

## 4. Data Augmentation
This resource expands the dataset by applying various transformations to the labeled images.

### Augment Labeled Dataset
This endpoint applies geometric and photometric filters to create synthetic variations of the training data.

- **Endpoint:** `/augment`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

#### Parameters
| Name | Type | Location | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `multiplier` | Integer | Body | Optional | Number of augmented versions per image (default: 5). |
| `geo_flip` | String | Body | Optional | Enable horizontal flips (`on` or empty). |
| `photo_bc` | String | Body | Optional | Enable brightness/contrast adjustments (`on` or empty). |
| `blur_gauss` | String | Body | Optional | Enable Gaussian blur (`on` or empty). |
| `noise_gauss` | String | Body | Optional | Enable Gaussian noise (`on` or empty). |

#### Request Example
```bash
curl -X POST http://localhost:5000/augment \
  -F "multiplier=3" \
  -F "geo_flip=on" \
  -F "photo_bc=on"
```

#### Response JSON Schema
```json
{
  "status": "success",
  "download_url": "/download/dataset_augmented.zip"
}
```

---

## 5. File Management
This resource provides access to the processed dataset packages.

### Download Dataset Package
This endpoint retrieves the generated `.zip` archives containing extraction, labeling, or augmentation results.

- **Endpoint:** `/download/<filename>`
- **Method:** `GET`

#### Request Example
```bash
curl -O http://localhost:5000/download/dataset_frames.zip
```



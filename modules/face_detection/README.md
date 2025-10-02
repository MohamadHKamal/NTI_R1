# Face Detection and Masking Module

The **face detection** module is the second stage of the privacy video de-identification pipeline. It performs robust face detection, quality assessment, and generates aligned face crops, masks, and comprehensive metadata using MediaPipe.

## Overview

This module processes videos to detect faces, select the highest quality frame, and generate all necessary outputs for subsequent synthetic face generation and video integration stages.

### Key Features

- 🎯 **Robust Face Detection**: MediaPipe FaceMesh for CPU-friendly, high-accuracy detection
- 📊 **Quality Assessment**: Advanced quality metrics for optimal frame selection
- 🖼️ **Face Alignment**: High-quality face crops with proper padding and alignment
- 🎭 **Mask Generation**: Binary and alpha masks for seamless face replacement
- 📍 **Landmark Detection**: 468-point facial landmarks with canonical keypoints
- 🎨 **Pose Estimation**: Head pose angles (yaw, pitch, roll) for better animation
- ⚡ **Smart Processing**: Early stopping and memory-efficient frame selection
- 📋 **Comprehensive Metadata**: Detailed JSON output with all processing information

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Dependencies

```bash
pip install opencv-python mediapipe numpy pandas
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Basic usage - process video and save to output directory
python modules/face_detection/detect_and_mask.py --input_video video.mp4 --out_dir output/

# Use with preprocessed job directory (recommended)
python modules/face_detection/detect_and_mask.py \
    --input_video "C:\path\to\video.mp4" \
    --out_dir "jobs/job_20250929_220934_e8d56915"

# High-resolution output with custom settings
python modules/face_detection/detect_and_mask.py \
    --input_video video.mp4 \
    --out_dir results/ \
    --out_size 512 \
    --sample_rate 2 \
    --min_detection_confidence 0.7

# Process limited frames for quick testing
python modules/face_detection/detect_and_mask.py \
    --input_video video.mp4 \
    --out_dir test_output/ \
    --max_frames 50 \
    --sample_rate 5
```

### Python API Usage

```python
from modules.face_detection.detect_and_mask import process_single_video, OutputPaths

# Setup output paths
video_name = "test_video"
output_paths = OutputPaths("output_directory", video_name)
output_paths.create_directories()

# Process video with default settings
result = process_single_video(
    video_path="path/to/video.mp4",
    output_paths=output_paths
)

# Process with custom parameters
result = process_single_video(
    video_path="video.mp4",
    output_paths=output_paths,
    output_size=512,
    sample_rate=2,
    min_detection_confidence=0.7,
    padding_ratio=0.2,
    max_frames=100,
    quality_threshold=0.2,
    verbose=True
)

# Check results
if result.status == 'ok':
    print(f"Best frame: {result.chosen_frame_index}")
    print(f"Face area: {result.chosen_bbox}")
else:
    print(f"Error: {result.error}")
```

## Parameters

### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input_video` | str | **Required** | Path to input video file |
| `--out_dir` | str | `single_video_output` | Output directory for all results |
| `--out_size` | int | 256 | Size for square face crops (e.g., 256, 512) |
| `--sample_rate` | int | 1 | Process every Nth frame (1=all frames, 2=every other frame) |
| `--min_detection_confidence` | float | 0.5 | MediaPipe detection confidence threshold (0.0-1.0) |
| `--pad` | float | 0.15 | Padding ratio around face bbox (0.1-0.3 recommended) |
| `--max_frames` | int | None | Maximum frames to process (None=process all) |
| `--quality_threshold` | float | 0.15 | Face area ratio threshold for early stopping |
| `--quiet` | flag | False | Suppress detailed progress output |

### Advanced Parameters

#### Face Quality Metrics

The module uses sophisticated quality assessment based on:

- **Face Size**: Larger faces score higher (normalized by image size)
- **Aspect Ratio**: Faces closer to square aspect ratio score higher
- **Edge Proximity**: Faces near image edges receive penalty
- **Detection Confidence**: MediaPipe confidence score
- **Landmark Quality**: Number and distribution of detected landmarks

#### Early Stopping Conditions

- **Max Consecutive No-Face**: Stops after 30 consecutive frames without faces
- **Quality Threshold**: Stops when excellent frame (quality > 0.8) is found
- **Frame Limit**: Stops at `max_frames` if specified

## Output Structure

The module generates a comprehensive set of outputs in the specified directory:

```
{out_dir}/
├── source_frames/
│   └── {video_name}_source.png        # Best quality full frame
├── aligned_crops/
│   └── {video_name}_face.png          # Square face crop (256x256 or custom size)
├── source_masks/
│   └── {video_name}_mask.png          # Binary face mask (full frame size)
├── masked_sources/
│   └── {video_name}_masked.png        # Full frame with face area blacked out
├── landmarks_json/
│   └── {video_name}_landmarks.json    # Detailed landmarks and metadata
└── {video_name}_summary.json          # Processing summary and statistics
```

## Output Files Details

### 1. Source Frame (`source_frames/{video_name}_source.png`)
- **Description**: The highest quality frame containing the detected face
- **Format**: PNG, original video resolution
- **Selection**: Based on combined face area and quality metrics
- **Usage**: Input for synthetic face generation modules

### 2. Aligned Face Crop (`aligned_crops/{video_name}_face.png`)
- **Description**: Square-cropped and padded face region
- **Format**: PNG, configurable size (default 256x256)
- **Features**: 
  - Proper aspect ratio maintenance
  - Padding around face for context
  - Square format for model compatibility
- **Usage**: Direct input for face generation models

### 3. Face Mask (`source_masks/{video_name}_mask.png`)
- **Description**: Binary mask defining face region in original frame
- **Format**: PNG, same size as source frame
- **Features**:
  - Elliptical face shape fitting
  - Smooth edges with Gaussian blur
  - High-quality landmark-based generation
- **Usage**: Face replacement and blending operations

### 4. Masked Source (`masked_sources/{video_name}_masked.png`)
- **Description**: Source frame with face region blacked out
- **Format**: PNG, same size as source frame
- **Usage**: Preview and validation of mask quality

### 5. Landmarks JSON (`landmarks_json/{video_name}_landmarks.json`)

Comprehensive metadata file containing:

```json
{
  "video_name": "test_video",
  "status": "ok",
  "fps": 30.0,
  "frame_count": 599,
  "schema_version": "1.0",
  "chosen_frame": {
    "index": 4,
    "name": "frame_000004",
    "timestamp_s": 0.1333,
    "bbox": [68, 58, 172, 193],
    "bbox_norm": [0.266, 0.227, 0.672, 0.754],
    "frame_path": "source_frames/test_video_source.png",
    "crop_path": "aligned_crops/test_video_face.png",
    "mask_path": "source_masks/test_video_mask.png"
  },
  "kps": {
    "left_eye": {"x": 85, "y": 102, "xn": 0.332, "yn": 0.399, "conf": 0.98},
    "right_eye": {"x": 142, "y": 105, "xn": 0.555, "yn": 0.410, "conf": 0.98},
    "nose_tip": {"x": 115, "y": 125, "xn": 0.449, "yn": 0.488, "conf": 0.99},
    "mouth_left": {"x": 98, "y": 152, "xn": 0.383, "yn": 0.594, "conf": 0.97},
    "mouth_right": {"x": 135, "y": 155, "xn": 0.527, "yn": 0.605, "conf": 0.97},
    "chin": {"x": 118, "y": 180, "xn": 0.461, "yn": 0.703, "conf": 0.96}
  },
  "pose": {
    "yaw": -2.1,
    "pitch": 5.3,
    "roll": -1.8,
    "pose_method": "solvePnP_v1"
  },
  "color_stats": {
    "mean_rgb": [142.5, 118.3, 103.7],
    "std_rgb": [45.2, 38.9, 41.1]
  },
  "summary": {
    "frames_processed": 11,
    "frames_with_faces": 11,
    "best_quality_score": 0.885,
    "avg_face_area": 13840.1
  }
}
```

### 6. Summary JSON (`{video_name}_summary.json`)

Processing statistics and frame analysis:

```json
{
  "video_info": {
    "path": "C:\\path\\to\\video.mp4",
    "name": "test_video",
    "frame_count": 599,
    "fps": 30.0,
    "dimensions": [256, 256]
  },
  "processing_result": {
    "status": "ok",
    "error": null,
    "chosen_frame_index": 4,
    "chosen_bbox": [68, 58, 172, 193]
  },
  "statistics": {
    "frames_processed": 11,
    "frames_with_faces": 11,
    "face_detection_rate": 1.0
  },
  "frame_analysis": {
    "total_frames_analyzed": 11,
    "frames_with_faces": 11,
    "quality_distribution": {
      "high_quality": 11,
      "medium_quality": 0,
      "low_quality": 0
    }
  }
}
```

## Supported Video Formats

- **MP4** (`.mp4`) - Recommended
- **MOV** (`.mov`)
- **AVI** (`.avi`)
- **MKV** (`.mkv`)
- **WebM** (`.webm`)

### Video Requirements

- **Resolution**: Minimum 64x64, faces should be at least 50x50 pixels
- **Face Size**: Recommended minimum 100x100 pixels for best quality
- **Lighting**: Well-lit faces produce better detection and quality scores
- **Angles**: Frontal to 3/4 view angles work best

## Quality Assessment

### Quality Metrics

The module evaluates face quality using multiple criteria:

1. **Size Score** (0.0-1.0): Based on face area relative to image size
2. **Aspect Score** (0.0-1.0): Preference for square-ish face bounding boxes
3. **Edge Penalty**: Reduction for faces near image edges
4. **Overall Quality**: Combined score used for frame selection

### Quality Categories

- **High Quality** (0.7-1.0): Excellent faces suitable for all applications
- **Medium Quality** (0.3-0.7): Good faces with minor issues
- **Low Quality** (0.0-0.3): Poor faces with significant problems

## Error Handling

### Common Error Types

| Status | Description | Common Causes |
|--------|-------------|---------------|
| `ok` | Processing completed successfully | - |
| `no_face` | No faces detected in video | Poor lighting, profile views, very small faces |
| `cannot_open_video` | Cannot read video file | Corrupted file, unsupported codec, permission issues |
| `fallback_center_crop` | Used center crop fallback | Invalid face detection, processing errors |
| `error` | General processing error | System issues, memory problems, invalid parameters |

### Troubleshooting

#### No Faces Detected

```bash
# Try lower confidence threshold
python modules/face_detection/detect_and_mask.py \
    --input_video video.mp4 \
    --out_dir output/ \
    --min_detection_confidence 0.3

# Process more frames
python modules/face_detection/detect_and_mask.py \
    --input_video video.mp4 \
    --out_dir output/ \
    --sample_rate 1 \
    --max_frames 200
```

#### Poor Quality Results

```bash
# Use higher resolution output
python modules/face_detection/detect_and_mask.py \
    --input_video video.mp4 \
    --out_dir output/ \
    --out_size 512

# Increase padding for more context
python modules/face_detection/detect_and_mask.py \
    --input_video video.mp4 \
    --out_dir output/ \
    --pad 0.25
```

## Performance Optimization

### Speed vs Quality Trade-offs

| Setting | Speed | Quality | Use Case |
|---------|-------|---------|----------|
| `--sample_rate 5 --max_frames 50` | Fastest | Good | Quick testing |
| `--sample_rate 2 --max_frames 100` | Fast | Better | Development |
| `--sample_rate 1` (default) | Slower | Best | Production |

### Memory Usage

- **Low Memory**: Use `--sample_rate 3` and `--max_frames 100`
- **High Quality**: Use `--sample_rate 1` and `--out_size 512`
- **Balanced**: Default settings work well for most use cases

## Integration with Pipeline

### Pipeline Sequence

1. **Preprocessing**: [`modules/preprocess/preprocess.py`](../preprocess/README.md)
2. **Face Detection**: `modules/face_detection/detect_and_mask.py` ← **Current**
3. **Synthetic Generation**: `modules/synthetic_face_generation/`

### Usage in Complete Pipeline

```bash
# Step 1: Preprocess video (generates job directory)
python modules/preprocess/preprocess.py --input_video video.mp4

# Step 2: Face detection (uses same job directory)
python modules/face_detection/detect_and_mask.py \
    --input_video video.mp4 \
    --out_dir jobs/{job_id_from_step1}

# Step 3: Continue with synthetic face generation...
```

### Output Compatibility

All outputs are designed for seamless integration:

- **Face crops**: Ready for BrushNet/Stable Diffusion input
- **Masks**: Compatible with OpenCV and PIL operations
- **Landmarks**: Standard format for animation models
- **Metadata**: JSON format for easy parsing by subsequent modules

## Examples

### Example 1: Quick Testing

```bash
python modules/face_detection/detect_and_mask.py \
    --input_video test.mp4 \
    --out_dir quick_test/ \
    --max_frames 20 \
    --sample_rate 3 \
    --quiet
```

### Example 2: High-Quality Production

```bash
python modules/face_detection/detect_and_mask.py \
    --input_video production_video.mp4 \
    --out_dir results/ \
    --out_size 512 \
    --min_detection_confidence 0.6 \
    --pad 0.2
```

### Example 3: Integration with Preprocessing

```bash
# Complete preprocessing + face detection
python modules/preprocess/preprocess.py --input_video video.mp4
# Note the job_id from output, then:
python modules/face_detection/detect_and_mask.py \
    --input_video video.mp4 \
    --out_dir jobs/job_20250929_220934_e8d56915
```

### Example 4: Batch Processing

```python
#!/usr/bin/env python3
import glob
import subprocess
from pathlib import Path

video_files = glob.glob("videos/*.mp4")

for video_file in video_files:
    video_name = Path(video_file).stem
    output_dir = f"output/{video_name}"
    
    cmd = [
        "python", "modules/face_detection/detect_and_mask.py",
        "--input_video", video_file,
        "--out_dir", output_dir,
        "--out_size", "512"
    ]
    
    print(f"Processing {video_file}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Success: {output_dir}")
    else:
        print(f"❌ Failed: {result.stderr}")
```

## MediaPipe Configuration

### Face Mesh Settings

The module uses MediaPipe FaceMesh with these settings:

- **Static Image Mode**: `True` (optimized for individual frames)
- **Max Faces**: `2` (processes up to 2 faces, selects largest)
- **Refine Landmarks**: `True` (468-point high-quality landmarks)
- **Min Detection Confidence**: Configurable (default 0.5)

### Landmark Indices

Key facial landmarks use MediaPipe's standard 468-point model:

- **Left Eye**: Index 33
- **Right Eye**: Index 263  
- **Nose Tip**: Index 1
- **Mouth Corners**: Indices 61, 291
- **Chin**: Index 18

## Advanced Features

### Canonical Keypoints

Extracts semantic keypoints with normalized coordinates:
- Eye centers, nose tip, mouth corners, chin
- Confidence scores for each point
- Both pixel and normalized (0-1) coordinates

### Head Pose Estimation

Estimates 3D head orientation:
- **Yaw**: Left-right rotation
- **Pitch**: Up-down rotation  
- **Roll**: Tilt rotation
- Method: Simplified solvePnP approach

### Color Statistics

Analyzes face crop colors:
- **Mean RGB**: Average color values
- **Standard Deviation**: Color variance
- Used for lighting and skin tone analysis

## Troubleshooting Guide

### Common Issues

1. **"No faces detected"**
   - Check lighting and face size
   - Lower `--min_detection_confidence`
   - Increase `--max_frames` and reduce `--sample_rate`

2. **Poor crop quality**
   - Increase `--out_size` to 512
   - Adjust `--pad` parameter
   - Check source video resolution

3. **Memory errors**
   - Use `--sample_rate 3` or higher
   - Set `--max_frames 100`
   - Process smaller videos

4. **Slow processing**
   - Increase `--sample_rate`
   - Set `--max_frames` limit
   - Use `--quiet` flag

### Performance Tips

- **CPU Usage**: MediaPipe is optimized for CPU processing
- **Memory**: Processing keeps only best frames in memory
- **Storage**: Each video generates ~1-5MB of output files
- **Speed**: Typical processing is 2-10x real-time depending on settings


## Future Improvements

When modifying this module:

1. Maintain output file format compatibility
2. Preserve landmark JSON schema
3. Update quality metrics carefully (affects frame selection)
4. Test with various video types and qualities
5. Update this README for any parameter changes

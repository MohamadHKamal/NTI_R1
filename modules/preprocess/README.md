# Video Preprocessing Module

The **preprocess** module is the first stage of the privacy video de-identification pipeline. It handles video validation, job initialization, audio extraction, and provenance tracking.

## Overview

This module validates input videos, creates unique job directories, extracts audio to standardized formats, and sets up the foundation for subsequent pipeline stages.

### Key Features

- ✅ **Video Validation**: Format, size, duration, and codec validation
- 📁 **Job Management**: Unique job ID generation and directory structure creation
- 🎵 **Audio Extraction**: High-quality WAV audio extraction using FFmpeg
- 📋 **Metadata Tracking**: Complete video metadata extraction and storage
- 🔄 **Provenance Logging**: Full pipeline step tracking in `meta.json`
- 📦 **Input Archival**: Safe archival of original video files

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio extraction and metadata)
- Virtual environment (recommended)

### Dependencies

```bash
pip install opencv-python-headless ffmpeg-python
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Basic usage with auto-generated job ID
python modules/preprocess/preprocess.py --input_video path/to/video.mp4

# Specify custom job ID
python modules/preprocess/preprocess.py --input_video video.mp4 --job_id custom_job_123

# Override maximum duration limit
python modules/preprocess/preprocess.py --input_video video.mp4 --max_duration 45

# Force overwrite existing job
python modules/preprocess/preprocess.py --input_video video.mp4 --overwrite
```

### Python API Usage

```python
from modules.preprocess.preprocess import VideoPreprocessor

# Initialize preprocessor
preprocessor = VideoPreprocessor()

# Process video with default settings
result = preprocessor.process_video("path/to/video.mp4")

# Process with custom parameters
result = preprocessor.process_video(
    input_video="video.mp4",
    job_id="custom_job",
    max_duration=30,
    audio_sample_rate=48000,
    overwrite=True
)

# Check results
if result["status"] == "success":
    print(f"Job created: {result['job_id']}")
    print(f"Job path: {result['job_path']}")
else:
    print(f"Error: {result['error']}")
```

## Parameters

### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input_video` | str | **Required** | Path to input video file |
| `--job_id` | str | Auto-generated | Custom job identifier (format: job_YYYYMMDD_HHMMSS_hash) |
| `--max_duration` | int | 30 | Maximum video duration in seconds |
| `--audio_sample_rate` | int | 44100 | Audio sample rate for extracted WAV |
| `--overwrite` | flag | False | Overwrite existing job directory if it exists |

### Supported Video Formats

- **MP4** (`.mp4`) - Recommended
- **MOV** (`.mov`)
- **AVI** (`.avi`)
- **MKV** (`.mkv`)
- **WebM** (`.webm`)

### Validation Criteria

- **File size**: Minimum 1KB, Maximum 2GB
- **Duration**: 1 second to `max_duration` (default 30s)
- **Resolution**: Minimum 64x64, Maximum 4096x4096
- **Codecs**: H.264, H.265, VP8, VP9, and other standard codecs

## Output Structure

The preprocessing module creates a standardized job directory structure:

```
jobs/{job_id}/
├── input/
│   └── {original_filename}     # Archived copy of input video
├── audio/
│   └── raw.wav                 # Extracted audio (44.1kHz, mono)
├── logs/
│   └── preprocess.log          # Processing logs
├── meta.json                   # Provenance tracking and job metadata
└── preprocess.json             # Detailed preprocessing results
```

## Output Files

### 1. `preprocess.json`
Detailed preprocessing results and video metadata:

```json
{
  "job_id": "job_20250929_220934_e8d56915",
  "status": "success",
  "created_at": "2025-09-29T22:09:34.652641",
  "processing_time_s": 0.16,
  "input_video": {
    "original_path": "C:\\path\\to\\video.mp4",
    "archived_path": "input/video.mp4",
    "filename": "video.mp4",
    "filesize_mb": 0.56,
    "duration_s": 19.97,
    "fps": 30.0,
    "frame_count": 599,
    "width": 256,
    "height": 256,
    "resolution": [256, 256],
    "codec": "h264",
    "bit_rate": 235627,
    "method": "ffmpeg"
  },
  "audio_extraction": {
    "success": true,
    "output_path": "jobs/job_id/audio/raw.wav",
    "sample_rate": 44100,
    "format": "wav",
    "channels": 1,
    "method": "ffmpeg"
  }
}
```

### 2. `meta.json`
Provenance tracking for the entire pipeline:

```json
{
  "job_id": "job_20250929_220934_e8d56915",
  "created_at": "2025-09-29T22:09:34.734147",
  "input_video": {
    "original_path": "C:\\path\\to\\video.mp4",
    "filename": "video.mp4"
  },
  "parameters": {
    "max_duration": 30,
    "audio_sample_rate": 44100,
    "overwrite": false
  },
  "steps": [
    {
      "step": "preprocess",
      "tool": "ffmpeg",
      "tool_version": "unknown",
      "params": {...},
      "runtime_s": 0.16,
      "outputs": ["input/video.mp4", "preprocess.json", "audio/raw.wav"],
      "status": "success",
      "timestamp": "2025-09-29T22:09:34.851664"
    }
  ]
}
```

### 3. `raw.wav`
Extracted audio file:
- **Format**: WAV (uncompressed)
- **Sample Rate**: 44.1kHz (configurable)
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit

## Error Handling

The module provides comprehensive error handling with descriptive messages:

### Common Error Types

| Error | Description | Solution |
|-------|-------------|----------|
| `file_not_found` | Input video file doesn't exist | Check file path and permissions |
| `unsupported_format` | Video format not supported | Convert to MP4, MOV, AVI, MKV, or WebM |
| `file_too_large` | Video exceeds size limit (2GB) | Compress or trim video |
| `duration_too_long` | Video exceeds max duration | Increase `--max_duration` or trim video |
| `invalid_video` | Corrupted or unreadable video | Check video file integrity |
| `audio_extraction_failed` | Cannot extract audio | Check FFmpeg installation |
| `job_exists` | Job directory already exists | Use `--overwrite` or choose different job_id |

### Example Error Output

```json
{
  "status": "error",
  "error": "duration_too_long",
  "error_details": "Video duration (45.2s) exceeds maximum allowed (30s)",
  "suggestions": ["Use --max_duration 50", "Trim video to 30 seconds"]
}
```

## Integration with Pipeline

This module is designed to integrate seamlessly with subsequent pipeline stages:

1. **Next Stage**: [`modules/face_detection/detect_and_mask.py`](../face_detection/README.md)
2. **Job Directory**: All outputs use the same job directory structure
3. **Metadata**: Subsequent modules append to `meta.json` for full provenance

### Usage in Pipeline

```bash
# Step 1: Preprocess video
python modules/preprocess/preprocess.py --input_video video.mp4

# Step 2: Use the job ID from preprocessing output
python modules/face_detection/detect_and_mask.py --input_video video.mp4 --out_dir jobs/{job_id}
```

## Performance Notes

- **Processing Speed**: Typically <1 second for videos under 30 seconds
- **Memory Usage**: Minimal - streams video for metadata extraction
- **Storage**: Creates copy of input video + extracted audio
- **FFmpeg**: Uses system FFmpeg for optimal performance

## Troubleshooting

### FFmpeg Issues

If you encounter FFmpeg-related errors:

1. **Install FFmpeg**:
   ```bash
   # Windows (using chocolatey)
   choco install ffmpeg
   
   # macOS (using homebrew)
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt install ffmpeg
   ```

2. **Verify Installation**:
   ```bash
   ffmpeg -version
   ```

3. **Path Issues**: Ensure FFmpeg is in your system PATH

### Common Solutions

- **Permission Errors**: Run with appropriate file permissions
- **Network Paths**: Copy files locally before processing
- **Unicode Paths**: Ensure proper encoding for international file names

## Examples

### Example 1: Basic Usage

```bash
python modules/preprocess/preprocess.py --input_video test_video.mp4
```

**Output**:
```
🎬 Starting video preprocessing
📁 Job ID: job_20250929_220934_e8d56915
🎥 Input: test_video.mp4
🔍 Validating input video...
✅ Video validated: 19.97s, 256x256, 0.6MB
📂 Creating job directory...
📋 Archiving input video...
✅ Video archived: jobs/job_20250929_220934_e8d56915/input/test_video.mp4
🎵 Extracting audio...
✅ Audio extracted: jobs/job_20250929_220934_e8d56915/audio/raw.wav
✅ Preprocessing complete!
📊 Processing time: 0.16s
📁 Job directory: jobs/job_20250929_220934_e8d56915
```

### Example 2: Custom Parameters

```bash
python modules/preprocess/preprocess.py \
    --input_video long_video.mp4 \
    --job_id my_custom_job \
    --max_duration 60 \
    --audio_sample_rate 48000 \
    --overwrite
```

### Example 3: Batch Processing Script

```python
#!/usr/bin/env python3
import glob
from modules.preprocess.preprocess import VideoPreprocessor

preprocessor = VideoPreprocessor()
video_files = glob.glob("videos/*.mp4")

for video_file in video_files:
    print(f"Processing {video_file}...")
    result = preprocessor.process_video(video_file)
    
    if result["status"] == "success":
        print(f"✅ Success: {result['job_id']}")
    else:
        print(f"❌ Failed: {result['error']}")
```


## Future Improvements

When modifying this module:

1. Maintain backward compatibility with existing job structures
2. Update `meta.json` format version if changing schema
3. Add comprehensive error handling for new features
4. Update this README with new parameters or features
5. Test with various video formats and edge cases
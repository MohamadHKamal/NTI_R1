# Data Collection & Preprocessing Pipeline

This notebook implements a comprehensive data collection and preprocessing pipeline for the project test dataset, specifically designed for the CelebV-HQ dataset. The pipeline filters, validates, and preprocesses video clips based on various criteria including duration, action labels, and face detection.

## Overview

The pipeline consists of several stages:
1. **Environment Setup** - Installing required dependencies
2. **Data Collection** - Filtering and selecting video clips from the dataset
3. **Data Preprocessing** - Validation, resizing, and quality checks
4. **Final Dataset Creation** - Generating the final processed dataset


## Pipeline Stages

### 1. Data Collection

The pipeline filters clips from the CelebV-HQ dataset based on:
- **Action filtering**: Only "talk" actions are included
- **Duration filtering**: Minimum clip length of 10 seconds
- **Bounding box filtering**: Minimum normalized area of 0.015
- **Uniqueness**: Selects the longest clip per unique source video

### 2. Audio Validation

Validates that selected clips contain audio tracks and filters out clips without audio.

### 3. Video Validation & Preprocessing

Performs comprehensive validation and preprocessing:
- **Face detection validation**: Ensures stable face presence using bounding boxes
- **Quality checks**: Minimum grayscale variance and center stability
- **Resizing**: Scales videos to 256x256 resolution while preserving aspect ratio
- **Format standardization**: Outputs H.264 encoded MP4 files

**Validation Parameters**:
- Sample frames: 20 per video
- Minimum box area fraction: 0.05
- Maximum center shift: 0.10
- Minimum region variance: 500.0

### 4. Final Dataset Creation

- Applies final duration filtering (≥10 seconds)
- Extracts audio tracks to separate files
- Generates comprehensive metadata CSV with action labels and duration information
- Creates downloadable dataset packages

## Filtering Results

The pipeline applies progressive filtering that significantly reduces the dataset size while maintaining quality:

1. **Initial Selection**: Started with ~2,600 videos that had "talk" action labels from the CelebV-HQ dataset
2. **Audio Filtering**: Filtered down to ~1,600 videos that contain valid audio streams
3. **Quality & Duration Filtering**: Applied face detection validation, stability checks, and minimum 10-second duration requirement
4. **Final Dataset**: Resulted in approximately ~700 high-quality videos with corresponding audio files

This multi-stage filtering ensures that only videos with stable face detection, clear audio, and sufficient duration make it into the final dataset, providing a curated collection suitable for machine learning applications.

## Output Structure

The pipeline generates:
- `videos_dataset/` - Processed video files (256x256, MP4 format)
- `audios_dataset/` - Extracted audio files (WAV format)
- `final_dataset_metadata.csv` - Complete metadata including clip IDs, durations, and action labels

## Key Features

- **Robust filtering**: Multi-stage filtering ensures high-quality dataset
- **Face stability validation**: Uses OpenCV for face detection and stability checks
- **Audio preservation**: Maintains original audio quality during video processing
- **Comprehensive logging**: Tracks failures and processing statistics at each stage
- **Quality metrics**: Provides detailed statistics on duration distribution and action labels

## Usage

Run the notebook cells sequentially. The pipeline is designed to be executed in a Google Colab environment with the CelebV-HQ dataset pre-downloaded.

## Output Statistics

The final section provides:
- File count and size statistics for each dataset component
- Duration distribution analysis
- Action label distribution visualization
- Downloadable ZIP archives of the final datasets

## Configuration

Key parameters can be adjusted in the configuration sections:
- `N_TARGET`: Target number of clips (default: 2600)
- `MIN_CLIP_SECONDS`: Minimum duration filter (default: 10.0)
- `MIN_AREA_NORM`: Minimum bounding box area (default: 0.015)
- `RES`: Output resolution (default: 256x256)

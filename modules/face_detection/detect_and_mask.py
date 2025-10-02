"""
Single Video Face Detection and Masking

A script for processing a single video file for face detection and preprocessing.
Based on the batch processing pipeline but optimized for single video analysis.

Output Structure:
    - source_frames/: Best full frame with detected face
    - aligned_crops/: Square face crop (256x256 default) 
    - source_masks/: Binary mask for face region
    - masked_sources/: Face crop with blacked-out face region
    - landmarks_json/: Facial landmarks and comprehensive metadata

Key Features:
    ✓ MediaPipe FaceMesh for robust CPU-friendly face detection
    ✓ Quality-based frame selection (size, aspect ratio, edge proximity)
    ✓ Real-time progress feedback for single video processing
    ✓ Detailed output with frame-by-frame analysis
    ✓ Memory-efficient processing with selective frame storage
    ✓ Comprehensive logging and error handling

Usage:
    python detect_and_mask.py --input_video path/to/video.mp4 --out_dir output/
    python detect_and_mask.py --input_video video.mp4 --out_dir results/ --out_size 512
    
Requirements:
    pip install opencv-python mediapipe numpy pandas
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Suppress MediaPipe warnings
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("mediapipe is required. Install with `pip install mediapipe`") from e

# Constants
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
MIN_FILE_SIZE_BYTES = 1024  # 1KB minimum
DEFAULT_OUTPUT_SIZE = 256
DEFAULT_PADDING_RATIO = 0.15
DEFAULT_BLUR_KERNEL_SIZE = 15
MAX_CONSECUTIVE_NO_FACE = 30
QUALITY_MEMORY_THRESHOLD = 0.8  # Keep frames within 80% of best quality

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SingleVideoResult:
    """Container for single video processing results and metadata."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.status = 'ok'
        self.error = None
        self.chosen_frame_index = None
        self.chosen_frame_name = None
        self.chosen_bbox = None
        self.frame_count = 0
        self.fps = None
        self.image_width = 0
        self.image_height = 0
        self.tracks = {}  # frame_idx -> {bbox, confidence, quality_metrics}
        self.landmarks = {}  # frame_idx -> list of [x,y] or None
        self.color_stats = None
        self.processing_stats = {}
    
    def set_error(self, error_msg: str, status: str = 'error'):
        """Set error status and message."""
        self.status = status
        self.error = error_msg
    
    def set_chosen_frame(self, frame_idx: int, bbox: List[int]):
        """Set the chosen frame information."""
        self.chosen_frame_index = frame_idx
        self.chosen_frame_name = f'frame_{frame_idx:06d}'
        self.chosen_bbox = bbox
    
    def add_processing_stats(self, stats: Dict):
        """Add processing statistics."""
        self.processing_stats.update(stats)


class OutputPaths:
    """Container for output file paths."""
    
    def __init__(self, base_output_dir: Union[str, Path], video_name: str):
        self.base_dir = Path(base_output_dir)
        self.video_name = video_name
        
        # Create subdirectories
        self.source_frames_dir = self.base_dir / 'source_frames'
        self.aligned_crops_dir = self.base_dir / 'aligned_crops'
        self.source_masks_dir = self.base_dir / 'source_masks'
        self.masked_sources_dir = self.base_dir / 'masked_sources'
        self.landmarks_json_dir = self.base_dir / 'landmarks_json'
        
        # File paths
        self.source_frame = self.source_frames_dir / f"{video_name}_source.png"
        self.aligned_crop = self.aligned_crops_dir / f"{video_name}_face.png"
        self.source_mask = self.source_masks_dir / f"{video_name}_mask.png"
        self.masked_source = self.masked_sources_dir / f"{video_name}_masked.png"
        self.landmarks_json = self.landmarks_json_dir / f"{video_name}_landmarks.json"
        self.summary_json = self.base_dir / f"{video_name}_summary.json"
    
    def create_directories(self):
        """Create all output directories."""
        for directory in [self.source_frames_dir, self.aligned_crops_dir, self.source_masks_dir, 
                         self.masked_sources_dir, self.landmarks_json_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS (copied from batch script)
# =============================================================================

def validate_video_file(video_path: Union[str, Path]) -> bool:
    """Validate that the video file exists and is accessible."""
    path = Path(video_path)
    
    # Check if file exists
    if not path.exists():
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Check file extension
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        print(f"Error: Unsupported video format: {path.suffix}")
        print(f"Supported formats: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}")
        return False
    
    # Check file size
    try:
        if path.stat().st_size < MIN_FILE_SIZE_BYTES:
            print(f"Error: Video file too small: {path}")
            return False
    except (OSError, IOError):
        print(f"Error: Cannot access video file: {path}")
        return False
    
    return True


def convert_landmarks_to_pixels(landmarks, image_width: int, image_height: int) -> List[List[int]]:
    """Convert normalized MediaPipe landmarks to pixel coordinates."""
    pixel_coords = []
    for landmark in landmarks:
        x = int(np.clip(landmark.x * image_width, 0, image_width - 1))
        y = int(np.clip(landmark.y * image_height, 0, image_height - 1))
        pixel_coords.append([x, y])
    return pixel_coords


def calculate_bounding_box(points: List[List[int]]) -> List[int]:
    """Calculate bounding box from a list of points."""
    if not points:
        return [0, 0, 0, 0]
    
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    
    return [
        int(min(x_coords)), int(min(y_coords)),
        int(max(x_coords)), int(max(y_coords))
    ]


def calculate_bbox_area(bbox: List[int]) -> int:
    """Calculate area of bounding box."""
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def _extract_canonical_keypoints(landmarks_px: List[List[int]], image_width: int, image_height: int) -> Dict[str, Dict[str, float]]:
    """Extract canonical semantic keypoints from MediaPipe landmarks with normalized coordinates and confidence."""
    if not landmarks_px or len(landmarks_px) < 468:
        return {}
    
    # MediaPipe face mesh canonical keypoint indices with typical confidence scores
    keypoint_indices = {
        'left_eye': (33, 0.98),      # Left eye center
        'right_eye': (263, 0.98),    # Right eye center
        'nose_tip': (1, 0.99),       # Nose tip
        'mouth_left': (61, 0.97),    # Left corner of mouth
        'mouth_right': (291, 0.97),  # Right corner of mouth
        'chin': (18, 0.96)           # Chin center
    }
    
    keypoints = {}
    for name, (idx, conf) in keypoint_indices.items():
        if idx < len(landmarks_px):
            x, y = landmarks_px[idx]
            # Calculate normalized coordinates (0-1 range)
            xn = round(x / image_width, 3) if image_width > 0 else 0.0
            yn = round(y / image_height, 3) if image_height > 0 else 0.0
            
            keypoints[name] = {
                "x": x,
                "y": y,
                "xn": xn,
                "yn": yn,
                "conf": conf
            }
    
    return keypoints


def _estimate_head_pose(landmarks_px: List[List[int]]) -> Dict[str, float]:
    """Estimate head pose angles from facial landmarks."""
    if not landmarks_px or len(landmarks_px) < 468:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "pose_method": "solvePnP_v1"}
    
    try:
        # Simple pose estimation using key facial landmarks
        left_eye = landmarks_px[33] if len(landmarks_px) > 33 else [0, 0]
        right_eye = landmarks_px[263] if len(landmarks_px) > 263 else [0, 0]
        nose_tip = landmarks_px[1] if len(landmarks_px) > 1 else [0, 0]
        chin = landmarks_px[18] if len(landmarks_px) > 18 else [0, 0]
        
        # Calculate roll (rotation around z-axis)
        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]
        roll = np.degrees(np.arctan2(eye_dy, eye_dx)) if eye_dx != 0 else 0.0
        
        # Estimate yaw and pitch (simplified)
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset_x = nose_tip[0] - face_center_x
        yaw = np.degrees(np.arctan2(nose_offset_x, 100)) * 0.5  # Rough approximation
        
        face_center_y = (left_eye[1] + right_eye[1]) / 2
        nose_chin_dy = chin[1] - nose_tip[1]
        pitch = np.degrees(np.arctan2(nose_tip[1] - face_center_y, nose_chin_dy)) * 0.5 if nose_chin_dy != 0 else 0.0
        
        return {
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1), 
            "roll": round(roll, 1),
            "pose_method": "solvePnP_v1"
        }
    except (ZeroDivisionError, IndexError):
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "pose_method": "solvePnP_v1"}


def extract_padded_crop(
    image: np.ndarray, 
    bbox: List[int], 
    padding_ratio: float = DEFAULT_PADDING_RATIO
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Extract a padded crop around the bounding box with comprehensive validation."""
    if image is None or image.size == 0:
        return None, None
        
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Validate and clamp bbox to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_width, x2), min(image_height, y2)
    
    if x1 >= x2 or y1 >= y2:
        return None, None
    
    # Calculate padding
    bbox_width = max(1, x2 - x1)
    bbox_height = max(1, y2 - y1)
    pad_width = int(bbox_width * padding_ratio)
    pad_height = int(bbox_height * padding_ratio)
    
    # Apply padding with bounds checking
    crop_x1 = max(0, x1 - pad_width)
    crop_y1 = max(0, y1 - pad_height)
    crop_x2 = min(image_width, x2 + pad_width)
    crop_y2 = min(image_height, y2 + pad_height)
    
    # Final validation
    if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
        return None, None
        
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_coords = (crop_x1, crop_y1, crop_x2, crop_y2)
    
    return crop, crop_coords


def create_face_mask_from_landmarks(
    landmarks_px: Optional[List[List[int]]], 
    crop_coordinates: Tuple[int, int, int, int], 
    output_size: int, 
    blur_kernel_size: int = DEFAULT_BLUR_KERNEL_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate binary and alpha masks from facial landmarks for cropped face region."""
    x1, y1, x2, y2 = crop_coordinates
    crop_width = max(1, x2 - x1)
    crop_height = max(1, y2 - y1)
    
    # Create base mask
    mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    
    if not landmarks_px or len(landmarks_px) == 0:
        mask = _create_elliptical_fallback_mask(crop_width, crop_height)
    else:
        # Convert landmarks to crop-relative coordinates
        crop_relative_points = []
        for x, y in landmarks_px:
            relative_x = int(x - x1)
            relative_y = int(y - y1)
            if 0 <= relative_x < crop_width and 0 <= relative_y < crop_height:
                crop_relative_points.append([relative_x, relative_y])
        
        if len(crop_relative_points) < 3:
            mask = _create_elliptical_fallback_mask(crop_width, crop_height)
        else:
            mask = _create_landmark_based_mask(crop_relative_points, crop_width, crop_height)
    
    # Generate smoothed masks
    return _generate_binary_and_alpha_masks(mask, output_size, blur_kernel_size)


def create_source_frame_mask_from_landmarks(
    landmarks_px: Optional[List[List[int]]], 
    frame_shape: Tuple[int, int],  # (height, width)
    crop_coordinates: Tuple[int, int, int, int],
    blur_kernel_size: int = DEFAULT_BLUR_KERNEL_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate high-quality face masks using ellipse fitting for natural face ovals.
    
    Args:
        landmarks_px: Facial landmarks in pixel coordinates (original frame coordinates)
        frame_shape: Shape of the original frame (height, width)
        crop_coordinates: Face crop region (x1, y1, x2, y2) for fallback
        blur_kernel_size: Gaussian blur kernel size
        
    Returns:
        Tuple of (binary_mask, alpha_mask) in original frame dimensions
    """
    frame_height, frame_width = frame_shape
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    if not landmarks_px or len(landmarks_px) == 0:
        # Fallback ellipse using crop coordinates
        x1, y1, x2, y2 = crop_coordinates
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        width, height = (x2 - x1) // 2, (y2 - y1) // 2
        cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)
    else:
        # Create face oval from key landmarks using ellipse fitting
        try:
            # Key face boundary landmarks (MediaPipe face mesh indices)
            # These correspond to the outer contour of the face
            key_landmark_indices = [
                10,   # Forehead center
                151,  # Forehead left
                337,  # Forehead right
                172,  # Left cheek
                397,  # Right cheek
                18,   # Chin center
                175,  # Jaw left
                400,  # Jaw right
                136,  # Left temple
                365,  # Right temple
                234,  # Left jaw curve
                454,  # Right jaw curve
                152,  # Nose bridge (for better face center)
                6,    # Nose tip
                54,   # Left mouth corner area
                284   # Right mouth corner area
            ]
            
            # Extract key landmarks, ensuring indices are valid
            key_landmarks = []
            for idx in key_landmark_indices:
                if idx < len(landmarks_px):
                    x, y = landmarks_px[idx]
                    # Ensure points are within frame bounds
                    x = max(0, min(frame_width - 1, int(x)))
                    y = max(0, min(frame_height - 1, int(y)))
                    key_landmarks.append([x, y])
            
            if len(key_landmarks) >= 5:  # Need at least 5 points for ellipse fitting
                points = np.array(key_landmarks, dtype=np.int32)
                
                # Fit ellipse to the key face landmarks
                ellipse = cv2.fitEllipse(points)
                
                # Extract ellipse parameters
                (center_x, center_y), (width, height), angle = ellipse
                
                # Make the ellipse slightly larger to ensure full face coverage
                width = int(width * 1.1)
                height = int(height * 1.1)
                
                # Draw the fitted ellipse
                cv2.ellipse(mask, (int(center_x), int(center_y)), (width//2, height//2), angle, 0, 360, 255, -1)
                
            elif len(key_landmarks) >= 3:
                # Fallback to convex hull if not enough points for ellipse
                points = np.array(key_landmarks, dtype=np.int32)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, 255)
            else:
                # Ultimate fallback to bbox ellipse
                x1, y1, x2, y2 = crop_coordinates
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                width, height = (x2 - x1) // 2, (y2 - y1) // 2
                cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)
                
        except (cv2.error, IndexError, ValueError, TypeError):
            # Fallback to bbox ellipse on any error
            x1, y1, x2, y2 = crop_coordinates
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            width, height = (x2 - x1) // 2, (y2 - y1) // 2
            cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)
    
    # Generate smoothed masks with enhanced smoothing
    return _generate_source_frame_masks(mask, blur_kernel_size)


def _create_elliptical_fallback_mask(width: int, height: int) -> np.ndarray:
    """Create elliptical mask as fallback when landmarks are insufficient."""
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    radius_x, radius_y = width // 3, height // 3
    cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
    return mask


def _create_landmark_based_mask(points: List[List[int]], width: int, height: int) -> np.ndarray:
    """Create mask from facial landmark points using convex hull."""
    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        hull = cv2.convexHull(np.array(points, dtype=np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    except cv2.error:
        # Fallback to elliptical mask if hull computation fails
        mask = _create_elliptical_fallback_mask(width, height)
    return mask


def _generate_binary_and_alpha_masks(
    mask: np.ndarray, 
    output_size: int, 
    blur_kernel_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate binary and alpha masks with proper smoothing for cropped regions."""
    # Ensure odd kernel size
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    
    # Apply Gaussian blur for smooth edges
    mask_blurred = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)
    
    # Create binary mask
    _, binary_mask = cv2.threshold(mask_blurred, 10, 255, cv2.THRESH_BINARY)
    
    # Create alpha mask (normalized to 0-1 range, then back to 0-255)
    alpha_mask = (mask_blurred.astype(np.float32) / 255.0 * 255).astype(np.uint8)
    
    # Resize both masks to output size
    binary_resized = cv2.resize(binary_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    alpha_resized = cv2.resize(alpha_mask, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    
    return binary_resized, alpha_resized


def _generate_source_frame_masks(
    mask: np.ndarray, 
    blur_kernel_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate high-quality binary and alpha masks with enhanced smoothing."""
    # Ensure larger kernel size for better smoothing on full-resolution images
    blur_kernel_size = max(blur_kernel_size, 25)  # Minimum 25x25 kernel for full-res images
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    
    # Apply morphological operations first to smooth the mask shape
    kernel_size = max(7, blur_kernel_size // 4)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close gaps and smooth boundaries
    mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
    
    # Apply strong Gaussian blur for very smooth edges
    mask_blurred = cv2.GaussianBlur(mask_smooth, (blur_kernel_size, blur_kernel_size), 0)
    
    # Create binary mask with higher threshold for cleaner edges
    _, binary_mask = cv2.threshold(mask_blurred, 50, 255, cv2.THRESH_BINARY)
    
    # Create alpha mask (keep the blurred version for smooth transitions)
    alpha_mask = mask_blurred.copy()
    
    return binary_mask, alpha_mask


def resize_to_square(image: np.ndarray, output_size: int) -> np.ndarray:
    """Resize image to square with padding, maintaining aspect ratio."""
    height, width = image.shape[:2]
    max_dimension = max(height, width)
    
    # Create square canvas
    canvas = np.zeros((max_dimension, max_dimension, 3), dtype=np.uint8)
    
    # Center the image on canvas
    y_offset = (max_dimension - height) // 2
    x_offset = (max_dimension - width) // 2
    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = image
    
    # Resize to target size
    resized = cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized


def calculate_color_statistics(image: np.ndarray) -> Dict[str, List[float]]:
    """Calculate color statistics for an image with reduced precision."""
    # Convert BGR to RGB for consistent color space
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Round to 2 decimal places to reduce JSON size
    mean_rgb = [round(x, 2) for x in rgb_image.mean(axis=(0, 1)).tolist()]
    std_rgb = [round(x, 2) for x in rgb_image.std(axis=(0, 1)).tolist()]
    
    return {
        'mean_rgb': mean_rgb,
        'std_rgb': std_rgb
    }


def assess_face_quality(
    bbox: List[int], 
    image_shape: Tuple[int, int], 
    landmarks_px: Optional[List[List[int]]] = None
) -> Dict[str, Union[float, bool, int]]:
    """Assess the quality of a detected face for frame selection."""
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Calculate basic face metrics
    face_width = x2 - x1
    face_height = y2 - y1
    face_area = face_width * face_height
    face_ratio = face_area / (image_width * image_height)
    aspect_ratio = face_width / max(face_height, 1)
    
    # Check edge proximity (faces near edges might be cropped)
    edge_margin_ratio = 0.05  # 5% margin
    edge_margin_x = image_width * edge_margin_ratio
    edge_margin_y = image_height * edge_margin_ratio
    
    near_edge = (
        x1 < edge_margin_x or y1 < edge_margin_y or 
        x2 > (image_width - edge_margin_x) or y2 > (image_height - edge_margin_y)
    )
    
    # Calculate quality scores
    size_score = min(face_ratio * 10, 1.0)  # Prefer larger faces, cap at 1.0
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 1.0) * 0.5)  # Prefer square-ish faces
    edge_penalty = 0.7 if near_edge else 1.0
    
    # Combined quality score
    overall_quality = size_score * aspect_score * edge_penalty
    
    return {
        'quality_score': overall_quality,
        'face_ratio': face_ratio,
        'aspect_ratio': aspect_ratio,
        'near_edge': near_edge,
        'face_area': face_area,
        'size_score': size_score,
        'aspect_score': aspect_score
    }


# =============================================================================
# MAIN VIDEO PROCESSING
# =============================================================================

def process_single_video(
    video_path: str, 
    output_paths: OutputPaths,
    output_size: int = DEFAULT_OUTPUT_SIZE,
    sample_rate: int = 1, 
    min_detection_confidence: float = 0.5, 
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    max_frames: Optional[int] = None, 
    quality_threshold: float = 0.15,
    verbose: bool = True
) -> SingleVideoResult:
    """
    Process a single video for face detection and preprocessing.
    
    Args:
        video_path: Path to input video file
        output_paths: Output file paths structure
        output_size: Target size for square face crops
        sample_rate: Process every Nth frame
        min_detection_confidence: Minimum MediaPipe detection confidence
        padding_ratio: Padding ratio for face crops  
        max_frames: Maximum frames to process (None for all)
        quality_threshold: Minimum face area ratio for early stopping
        verbose: Print detailed progress information
        
    Returns:
        SingleVideoResult with processing status and metadata
    """
    result = SingleVideoResult(video_path)

    try:
        if verbose:
            print(f"🎬 Opening video: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result.set_error('cannot_open_video')
            return result
        
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        result.fps = fps
        result.frame_count = frame_count
        
        if verbose:
            print(f"📊 Video info: {frame_count} frames @ {fps:.1f} fps")
            if max_frames:
                print(f"🎯 Processing max {max_frames} frames (sample rate: 1/{sample_rate})")

        face_areas = []  # (frame_idx, area, bbox, landmarks_px, frame_img)
        best_face_area = 0
        early_stop_threshold = quality_threshold * 640 * 480  # Default threshold

        with mp_face_mesh.FaceMesh(static_image_mode=True,
                                   max_num_faces=2,
                                   refine_landmarks=True,
                                   min_detection_confidence=min_detection_confidence) as face_mesh:

            frame_idx = 0
            read_idx = 0
            consecutive_no_face = 0
            faces_found = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                if sample_rate > 1 and (frame_idx - 1) % sample_rate != 0:
                    continue
                    
                read_idx += 1
                
                # Early stopping conditions
                if max_frames and read_idx > max_frames:
                    if verbose:
                        print(f"⏹️ Stopped at max frames limit: {max_frames}")
                    break
                if consecutive_no_face > MAX_CONSECUTIVE_NO_FACE and len(face_areas) > 0:
                    if verbose:
                        print(f"⏹️ Early stop: {MAX_CONSECUTIVE_NO_FACE} consecutive frames without faces")
                    break

                h, w = frame.shape[:2]
                # Store image dimensions on first frame
                if result.image_width == 0:
                    result.image_width = w
                    result.image_height = h
                    early_stop_threshold = quality_threshold * w * h
                
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_mp = face_mesh.process(img_rgb)
                key = f'frame_{frame_idx:06d}'

                if not results_mp.multi_face_landmarks:
                    result.landmarks[key] = None
                    result.tracks[key] = {'bbox': None, 'confidence': 0.0}
                    consecutive_no_face += 1
                    if verbose and read_idx % 50 == 0:
                        print(f"⏳ Frame {read_idx}: No faces found")
                    continue

                consecutive_no_face = 0
                faces_found += 1
                
                # Handle multiple faces - pick the largest one
                best_face_lms = None
                best_area = 0
                for face_lms in results_mp.multi_face_landmarks:
                    pts_px = convert_landmarks_to_pixels(face_lms.landmark, w, h)
                    bbox = calculate_bounding_box(pts_px)
                    area = calculate_bbox_area(bbox)
                    if area > best_area:
                        best_area = area
                        best_face_lms = face_lms

                if best_face_lms is None:
                    continue

                pts_px = convert_landmarks_to_pixels(best_face_lms.landmark, w, h)
                bbox = calculate_bounding_box(pts_px)
                area = calculate_bbox_area(bbox)
                confidence = float(area) / float(max(1, w * h))
                
                # Assess face quality
                quality_metrics = assess_face_quality(bbox, (h, w), pts_px)
                quality_score = quality_metrics['quality_score']
                
                # Only store landmarks for high-quality frames to save memory
                if quality_score > 0.5:  # Only store landmarks for good frames
                    result.landmarks[key] = pts_px
                else:
                    result.landmarks[key] = None
                
                # Store minimal tracking info
                result.tracks[key] = {
                    'bbox': bbox, 
                    'confidence': round(confidence, 3),
                    'quality_score': round(quality_score, 3),
                    'face_area': area
                }
                
                # Only store frame copy if this could be the best frame (memory optimization)
                combined_score = area * quality_score
                if combined_score > best_face_area * QUALITY_MEMORY_THRESHOLD:
                    face_areas.append((frame_idx, combined_score, bbox, pts_px, frame.copy(), quality_score))
                    best_face_area = max(best_face_area, combined_score)
                
                if verbose and (read_idx % 25 == 0 or quality_score > 0.8):
                    print(f"✨ Frame {read_idx}: Face detected! Area={area}px², Quality={quality_score:.3f}")
                    
                    # Early stopping if we found a very good frame
                    if quality_score > 0.8 and confidence > quality_threshold:
                        if len(face_areas) > 10 and combined_score > early_stop_threshold:
                            if verbose:
                                print(f"🎯 Found excellent frame {frame_idx} - stopping early")
                            break

        cap.release()
        
        # Add processing statistics
        result.add_processing_stats({
            'frames_processed': read_idx,
            'frames_with_faces': faces_found,
            'face_detection_rate': round(faces_found / max(1, read_idx), 3)
        })

        if verbose:
            print(f"📈 Processing complete: {read_idx} frames processed, {faces_found} with faces")

        # Choose best frame (highest combined score: area * quality)
        if len(face_areas) > 0:
            face_areas_sorted = sorted(face_areas, key=lambda x: x[1], reverse=True)
            chosen = face_areas_sorted[0]
            chosen_frame_idx, chosen_score, chosen_bbox, chosen_landmarks_px, chosen_img, chosen_quality = chosen
            result.set_chosen_frame(chosen_frame_idx, chosen_bbox)
            
            if verbose:
                print(f"🏆 Best frame: {chosen_frame_idx} (quality: {chosen_quality:.3f}, area: {calculate_bbox_area(chosen_bbox)}px²)")
        else:
            # No faces found; create fallback
            result.set_error('no_face_detected', 'no_face')
            if verbose:
                print("❌ No faces detected in video")
            
            # Try to read first frame as a thumbnail
            cap2 = cv2.VideoCapture(str(video_path))
            ret2, frame0 = cap2.read()
            cap2.release()
            if ret2:
                thumb = resize_to_square(frame0, output_size)
                cv2.imwrite(str(output_paths.source_frame), thumb)
                if verbose:
                    print(f"💾 Saved fallback thumbnail: {output_paths.source_frame}")
            return result

        # Create crop and mask for chosen frame
        crop_img, crop_box = extract_padded_crop(chosen_img, chosen_bbox, padding_ratio)
        
        # Fallback if crop empty or invalid
        if crop_img is None or crop_img.size == 0:
            # Try direct bbox crop as fallback
            x1, y1, x2, y2 = chosen_bbox
            h, w = chosen_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x1 < x2 and y1 < y2:
                crop_img = chosen_img[y1:y2, x1:x2].copy()
                crop_box = (x1, y1, x2, y2)
            else:
                # Last resort: use center crop
                center_size = min(h, w) // 2
                cy, cx = h // 2, w // 2
                y1 = max(0, cy - center_size // 2)
                y2 = min(h, cy + center_size // 2)
                x1 = max(0, cx - center_size // 2)
                x2 = min(w, cx + center_size // 2)
                crop_img = chosen_img[y1:y2, x1:x2].copy()
                crop_box = (x1, y1, x2, y2)
                result.status = 'fallback_center_crop'

        if verbose:
            print("💾 Saving output files...")

        # Save full source frame (chosen frame)
        cv2.imwrite(str(output_paths.source_frame), chosen_img)

        # Save aligned crop (square resize)
        face_crop = resize_to_square(crop_img, output_size)
        cv2.imwrite(str(output_paths.aligned_crop), face_crop)

        # Generate masks from FULL SOURCE FRAME (not crop)
        frame_shape = (chosen_img.shape[0], chosen_img.shape[1])  # (height, width)
        source_binary_mask, source_alpha_mask = create_source_frame_mask_from_landmarks(
            chosen_landmarks_px, frame_shape, crop_box
        )
        cv2.imwrite(str(output_paths.source_mask), source_binary_mask)

        # Generate masked source from FULL SOURCE FRAME (face area blacked out)
        masked_source = chosen_img.copy()
        masked_source[source_binary_mask == 255] = (0, 0, 0)
        cv2.imwrite(str(output_paths.masked_source), masked_source)

        # Calculate color stats for crop
        result.color_stats = calculate_color_statistics(face_crop)

        # Save landmarks JSON
        _save_landmarks_json(result, output_paths)
        
        # Save detailed summary
        _save_summary_json(result, output_paths)
        
        if verbose:
            print(f"✅ Processing complete! Files saved to: {output_paths.base_dir}")
            print(f"📄 Summary: {output_paths.summary_json}")
        
        return result

    except Exception as e:
        result.set_error(str(e))
        if verbose:
            print(f"❌ Error processing video: {e}")
        return result


def _save_landmarks_json(result: SingleVideoResult, output_paths: OutputPaths):
    """Save landmarks and metadata to JSON file in the specified format."""
    
    # Get actual image dimensions for normalization
    image_width = result.image_width if result.image_width > 0 else 640
    image_height = result.image_height if result.image_height > 0 else 480
    
    # Get canonical keypoints and pose for chosen frame
    kps = {}
    pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "pose_method": "solvePnP_v1"}
    
    if result.chosen_frame_name and result.chosen_frame_name in result.landmarks:
        full_landmarks = result.landmarks[result.chosen_frame_name]
        if full_landmarks:
            kps = _extract_canonical_keypoints(full_landmarks, image_width, image_height)
            pose = _estimate_head_pose(full_landmarks)
    
    # Calculate timestamp for chosen frame
    timestamp_s = 0.0
    if result.chosen_frame_index is not None and result.fps:
        timestamp_s = round(result.chosen_frame_index / result.fps, 4)
    
    # Calculate normalized bbox coordinates
    bbox_norm = [0.0, 0.0, 0.0, 0.0]
    if result.chosen_bbox and image_width > 0 and image_height > 0:
        x1, y1, x2, y2 = result.chosen_bbox
        bbox_norm = [
            round(x1 / image_width, 3),
            round(y1 / image_height, 3),
            round(x2 / image_width, 3),
            round(y2 / image_height, 3)
        ]
    
    # Build file paths (relative to output directory)
    frame_path = f"source_frames/{result.video_name}_source.png"
    crop_path = f"aligned_crops/{result.video_name}_face.png"
    mask_path = f"source_masks/{result.video_name}_mask.png"
    
    # Build the exact format requested
    metadata = {
        "video_name": result.video_name,
        "status": result.status,
        "fps": round(result.fps, 2) if result.fps else None,
        "frame_count": result.frame_count,
        "schema_version": "1.0",
        "chosen_frame": {
            "index": result.chosen_frame_index,
            "name": result.chosen_frame_name,
            "timestamp_s": timestamp_s,
            "bbox": result.chosen_bbox,
            "bbox_norm": bbox_norm,
            "frame_path": frame_path,
            "crop_path": crop_path,
            "mask_path": mask_path
        },
        "kps": kps,
        "pose": pose,
        "color_stats": result.color_stats,
        "summary": {
            "frames_processed": len(result.tracks),
            "frames_with_faces": len([k for k, v in result.tracks.items() if v['bbox'] is not None]),
            "best_quality_score": round(max([v.get('quality_score', 0) for v in result.tracks.values()], default=0), 3),
            "avg_face_area": round(sum([v.get('face_area', 0) for v in result.tracks.values() if v.get('face_area', 0) > 0]) / 
                                 max(1, len([v for v in result.tracks.values() if v.get('face_area', 0) > 0])), 1)
        }
    }
    
    # Clean up None values and add error if present
    if result.error:
        metadata["error"] = result.error
    
    # Remove None values
    if metadata["fps"] is None:
        del metadata["fps"]
        
    with open(output_paths.landmarks_json, 'w') as f:
        json.dump(metadata, f, indent=2)


def _save_summary_json(result: SingleVideoResult, output_paths: OutputPaths):
    """Save detailed processing summary to JSON file."""
    
    summary = {
        "video_info": {
            "path": result.video_path,
            "name": result.video_name,
            "frame_count": result.frame_count,
            "fps": result.fps,
            "dimensions": [result.image_width, result.image_height]
        },
        "processing_result": {
            "status": result.status,
            "error": result.error,
            "chosen_frame_index": result.chosen_frame_index,
            "chosen_bbox": result.chosen_bbox
        },
        "statistics": result.processing_stats,
        "frame_analysis": {
            "total_frames_analyzed": len(result.tracks),
            "frames_with_faces": len([k for k, v in result.tracks.items() if v['bbox'] is not None]),
            "quality_distribution": {
                "high_quality": len([v for v in result.tracks.values() if v.get('quality_score', 0) > 0.7]),
                "medium_quality": len([v for v in result.tracks.values() if 0.3 < v.get('quality_score', 0) <= 0.7]),
                "low_quality": len([v for v in result.tracks.values() if 0 < v.get('quality_score', 0) <= 0.3])
            }
        },
        "output_files": {
            "source_frame": str(output_paths.source_frame),
            "aligned_crop": str(output_paths.aligned_crop),
            "source_mask": str(output_paths.source_mask),
            "masked_source": str(output_paths.masked_source),
            "landmarks_json": str(output_paths.landmarks_json)
        }
    }
    
    with open(output_paths.summary_json, 'w') as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Single video face detection & mask generation using MediaPipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input_video', required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--out_dir', default='single_video_output',
        help='Output directory for processed results'
    )
    parser.add_argument(
        '--out_size', type=int, default=DEFAULT_OUTPUT_SIZE,
        help='Size for square face crops'
    )
    parser.add_argument(
        '--sample_rate', type=int, default=1,
        help='Process every Nth frame (1 = process all frames)'
    )
    parser.add_argument(
        '--min_detection_confidence', type=float, default=0.5,
        help='Minimum MediaPipe detection confidence threshold'
    )
    parser.add_argument(
        '--pad', type=float, default=DEFAULT_PADDING_RATIO,
        help='Padding ratio for face crops'
    )
    parser.add_argument(
        '--max_frames', type=int, default=None,
        help='Maximum frames to process (None = process all)'
    )
    parser.add_argument(
        '--quality_threshold', type=float, default=0.15,
        help='Minimum face area ratio for early stopping'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress detailed output messages'
    )
    
    return parser


def main():
    """Main entry point for single video face detection."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate input video
    if not validate_video_file(args.input_video):
        return 1

    # Setup output paths
    video_name = Path(args.input_video).stem
    output_paths = OutputPaths(args.out_dir, video_name)
    output_paths.create_directories()
    
    verbose = not args.quiet
    
    if verbose:
        print("🎬 Single Video Face Detection & Masking Pipeline")
        print("=" * 60)
        print(f"📁 Input video: {args.input_video}")
        print(f"📂 Output directory: {output_paths.base_dir}")
        print(f"⚙️ Settings: size={args.out_size}, sample_rate={args.sample_rate}")
        if args.max_frames:
            print(f"🎯 Max frames: {args.max_frames}")
        print("-" * 60)

    # Process the video
    result = process_single_video(
        video_path=args.input_video,
        output_paths=output_paths,
        output_size=args.out_size,
        sample_rate=args.sample_rate,
        min_detection_confidence=args.min_detection_confidence,
        padding_ratio=args.pad,
        max_frames=args.max_frames,
        quality_threshold=args.quality_threshold,
        verbose=verbose
    )
    
    # Print final results
    if verbose:
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)
        print(f"Status: {result.status}")
        if result.chosen_frame_index is not None:
            print(f"Best frame: #{result.chosen_frame_index} @ {result.chosen_frame_index/result.fps:.2f}s")
            if result.chosen_bbox:
                area = calculate_bbox_area(result.chosen_bbox)
                print(f"Face area: {area}px²")
        
        if result.error:
            print(f"Error: {result.error}")
            
        print(f"\n📄 Detailed summary: {output_paths.summary_json}")
        print(f"🎨 Landmarks data: {output_paths.landmarks_json}")

    # Return appropriate exit code
    return 0 if result.status in ['ok', 'fallback_center_crop'] else 1


if __name__ == '__main__':
    exit(main())
"""
Video Preprocessing Pipeline Module

First stage of the privacy video de-identification pipeline that handles:
- Job directory initialization with unique IDs
- Video validation and metadata extraction
- Audio extraction to standardized WAV format
- Input video archival and provenance tracking

Architecture:
    Input Video → Validation → Job Creation → Audio Extraction → Results

Usage:
    python -m modules.preprocess.preprocess --input_video video.mp4
    python -m modules.preprocess.preprocess --input_video video.mp4 --job_id custom_123

Requirements:
    - opencv-python
    - ffmpeg (audio extraction and metadata)

Author: Privacy Video De-identification Pipeline
Version: 2.0
"""

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import cv2


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

class Config:
    """Centralized configuration for the preprocessing module."""
    
    # File validation
    SUPPORTED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
    MIN_FILE_SIZE_BYTES = 1024  # 1KB
    MAX_FILE_SIZE_MB = 500  # 500MB
    
    # Processing defaults
    DEFAULT_MAX_DURATION = 30  # seconds
    DEFAULT_AUDIO_SAMPLE_RATE = 44100
    DEFAULT_JOBS_DIR = "jobs"
    
    # Job structure
    JOB_DIRECTORIES = ["input", "audio", "logs", "preview"]
    
    # Audio extraction
    AUDIO_FORMAT = "wav"
    AUDIO_CODEC = "pcm_s16le"  # PCM 16-bit
    AUDIO_CHANNELS = 1  # Mono


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class Utils:
    """Utility functions for video processing and system operations."""
    
    @staticmethod
    def generate_job_id() -> str:
        """Generate unique job ID with timestamp and UUID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"job_{timestamp}_{short_uuid}"
    
    @staticmethod
    def check_ffmpeg_available() -> bool:
        """Check if FFmpeg is available in system PATH."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    @staticmethod
    def safe_eval_fraction(fraction_str: str) -> float:
        """Safely evaluate fraction string (e.g., '30/1' -> 30.0)."""
        try:
            if '/' in fraction_str:
                numerator, denominator = fraction_str.split('/')
                return float(numerator) / float(denominator) if float(denominator) != 0 else 0.0
            return float(fraction_str)
        except (ValueError, ZeroDivisionError):
            return 0.0


# =============================================================================
# VIDEO ANALYSIS
# =============================================================================

class VideoAnalyzer:
    """Handles video metadata extraction using FFmpeg and OpenCV fallback."""
    
    @staticmethod
    def get_info_ffmpeg(video_path: str) -> Dict[str, Any]:
        """Extract comprehensive video metadata using FFprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return {"error": f"FFprobe failed: {result.stderr}"}
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
                None
            )
            
            if not video_stream:
                return {"error": "No video stream found"}
            
            format_info = data.get('format', {})
            
            return {
                "duration_s": round(float(format_info.get('duration', 0)), 2),
                "fps": round(Utils.safe_eval_fraction(video_stream.get('r_frame_rate', '0/1')), 2),
                "frame_count": int(video_stream.get('nb_frames', 0)),
                "width": int(video_stream.get('width', 0)),
                "height": int(video_stream.get('height', 0)),
                "resolution": [int(video_stream.get('width', 0)), int(video_stream.get('height', 0))],
                "codec": video_stream.get('codec_name', 'unknown'),
                "bit_rate": int(format_info.get('bit_rate', 0)),
                "method": "ffmpeg"
            }
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            return {"error": f"FFmpeg analysis failed: {str(e)}"}
    
    @staticmethod
    def get_info_opencv(video_path: str) -> Dict[str, Any]:
        """Extract basic video metadata using OpenCV (fallback)."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {"error": "Cannot open video with OpenCV"}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "duration_s": round(duration, 2),
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "resolution": [width, height],
                "codec": "unknown",  # OpenCV limitation
                "method": "opencv"
            }
        except Exception as e:
            return {"error": f"OpenCV analysis failed: {str(e)}"}
    
    @classmethod
    def get_video_info(cls, video_path: str) -> Dict[str, Any]:
        """Get video metadata using best available method."""
        if Utils.check_ffmpeg_available():
            return cls.get_info_ffmpeg(video_path)
        else:
            print("⚠️  Warning: FFmpeg not found, using OpenCV for metadata (limited info)")
            return cls.get_info_opencv(video_path)


# =============================================================================
# AUDIO EXTRACTION
# =============================================================================

class AudioExtractor:
    """Handles audio extraction from video files."""
    
    @staticmethod
    def extract_with_ffmpeg(video_path: str, output_path: Path, sample_rate: int) -> Dict[str, Any]:
        """Extract audio using FFmpeg with standardized format."""
        try:
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn',  # No video
                '-acodec', Config.AUDIO_CODEC,
                '-ar', str(sample_rate),
                '-ac', str(Config.AUDIO_CHANNELS),
                '-y',  # Overwrite output
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"FFmpeg audio extraction failed: {result.stderr}",
                    "method": "ffmpeg"
                }
            
            # Validate output file
            if output_path.exists() and output_path.stat().st_size > 1000:  # Minimum 1KB
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "sample_rate": sample_rate,
                    "format": Config.AUDIO_FORMAT,
                    "channels": Config.AUDIO_CHANNELS,
                    "method": "ffmpeg"
                }
            else:
                return {
                    "success": False,
                    "error": "Audio file not created or too small",
                    "method": "ffmpeg"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "FFmpeg audio extraction timed out",
                "method": "ffmpeg"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Audio extraction error: {str(e)}",
                "method": "ffmpeg"
            }
    
    @staticmethod
    def extract_fallback() -> Dict[str, Any]:
        """Return error for systems without FFmpeg."""
        return {
            "success": False,
            "error": "OpenCV cannot extract audio. Install FFmpeg for audio extraction.",
            "method": "opencv_fallback"
        }
    
    @classmethod
    def extract_audio(cls, video_path: str, output_path: Path, sample_rate: int) -> Dict[str, Any]:
        """Extract audio using best available method."""
        if Utils.check_ffmpeg_available():
            return cls.extract_with_ffmpeg(video_path, output_path, sample_rate)
        else:
            return cls.extract_fallback()


# =============================================================================
# VIDEO VALIDATION
# =============================================================================

class VideoValidator:
    """Handles video file validation and basic metadata extraction."""
    
    @staticmethod
    def validate_file(video_path: str, max_duration: float, max_size_mb: float) -> Dict[str, Any]:
        """Comprehensive video file validation."""
        path = Path(video_path)
        
        # Check existence
        if not path.exists():
            return {"valid": False, "error": f"Video file not found: {video_path}"}
        
        # Check extension
        if path.suffix.lower() not in Config.SUPPORTED_EXTENSIONS:
            supported = ', '.join(Config.SUPPORTED_EXTENSIONS)
            return {
                "valid": False,
                "error": f"Unsupported format: {path.suffix}. Supported: {supported}"
            }
        
        # Check file size
        try:
            file_size_bytes = path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            if file_size_bytes < Config.MIN_FILE_SIZE_BYTES:
                return {"valid": False, "error": f"File too small: {file_size_bytes} bytes"}
            
            if file_size_mb > max_size_mb:
                return {"valid": False, "error": f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"}
                
        except (OSError, IOError) as e:
            return {"valid": False, "error": f"Cannot access file: {e}"}
        
        # Analyze video metadata
        video_info = VideoAnalyzer.get_video_info(video_path)
        
        if "error" in video_info:
            return {"valid": False, "error": f"Video analysis failed: {video_info['error']}"}
        
        # Check duration
        duration = video_info.get("duration_s", 0)
        if duration > max_duration:
            return {
                "valid": False,
                "error": f"Video too long: {duration:.1f}s (max: {max_duration}s)"
            }
        
        return {
            "valid": True,
            "file_size_mb": file_size_mb,
            "video_info": video_info
        }


# =============================================================================
# JOB MANAGEMENT
# =============================================================================

class JobManager:
    """Handles job directory creation and metadata management."""
    
    @staticmethod
    def create_job_directory(job_id: str, base_dir: str) -> Path:
        """Create job directory structure."""
        jobs_dir = Path(base_dir)
        jobs_dir.mkdir(exist_ok=True)
        
        job_path = jobs_dir / job_id
        job_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in Config.JOB_DIRECTORIES:
            (job_path / subdir).mkdir(exist_ok=True)
        
        return job_path
    
    @staticmethod
    def initialize_metadata(job_id: str, input_video: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize job metadata with provenance tracking."""
        return {
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "input_video": {
                "original_path": str(input_video),
                "filename": Path(input_video).name
            },
            "parameters": params,
            "steps": []
        }
    
    @staticmethod
    def update_metadata(job_path: Path, step_info: Dict[str, Any]) -> None:
        """Update job metadata with processing step."""
        meta_file = job_path / "meta.json"
        
        # Load existing metadata
        metadata = {"steps": []}
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
        
        # Add step with timestamp
        step_info["timestamp"] = datetime.now().isoformat()
        metadata["steps"].append(step_info)
        
        # Save updated metadata
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def save_results(job_path: Path, results: Dict[str, Any]) -> None:
        """Save preprocessing results to preprocess.json."""
        with open(job_path / "preprocess.json", 'w') as f:
            json.dump(results, f, indent=2)


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

class VideoPreprocessor:
    """Main video preprocessing pipeline coordinator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize preprocessor with optional configuration override."""
        self.config = config or {}
    
    def process(
        self,
        input_video: str,
        job_id: Optional[str] = None,
        max_duration: float = Config.DEFAULT_MAX_DURATION,
        audio_sample_rate: int = Config.DEFAULT_AUDIO_SAMPLE_RATE,
        jobs_dir: str = Config.DEFAULT_JOBS_DIR,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete video preprocessing pipeline.
        
        Args:
            input_video: Path to input video file
            job_id: Unique job identifier (auto-generated if None)
            max_duration: Maximum video duration in seconds
            audio_sample_rate: Audio extraction sample rate
            jobs_dir: Base directory for job storage
            overwrite: Whether to overwrite existing job
            
        Returns:
            Processing results dictionary with status and metadata
        """
        start_time = datetime.now()
        
        # Generate job ID if needed
        if job_id is None:
            job_id = Utils.generate_job_id()
        
        self._print_header(job_id, input_video)
        
        # Validate input video
        print("🔍 Validating input video...")
        validation = VideoValidator.validate_file(input_video, max_duration, Config.MAX_FILE_SIZE_MB)
        
        if not validation["valid"]:
            return self._create_error_result(validation["error"], job_id)
        
        video_info = validation["video_info"]
        file_size_mb = validation["file_size_mb"]
        self._print_validation_success(video_info, file_size_mb)
        
        # Create job directory
        print("📂 Creating job directory...")
        try:
            job_path = JobManager.create_job_directory(job_id, jobs_dir)
            
            if (job_path / "preprocess.json").exists() and not overwrite:
                return self._create_error_result(
                    f"Job {job_id} already exists. Use --overwrite to replace.",
                    job_id
                )
        except Exception as e:
            return self._create_error_result(f"Failed to create job directory: {e}", job_id)
        
        # Initialize and save metadata
        params = {
            "max_duration": max_duration,
            "audio_sample_rate": audio_sample_rate,
            "overwrite": overwrite
        }
        metadata = JobManager.initialize_metadata(job_id, input_video, params)
        
        with open(job_path / "meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Archive input video
        print("📋 Archiving input video...")
        try:
            input_copy_path = job_path / "input" / Path(input_video).name
            shutil.copy2(input_video, input_copy_path)
            print(f"✅ Video archived: {input_copy_path}")
        except Exception as e:
            return self._create_error_result(f"Failed to copy input video: {e}", job_id)
        
        # Extract audio
        print("🎵 Extracting audio...")
        audio_output_path = job_path / "audio" / f"raw.{Config.AUDIO_FORMAT}"
        audio_result = AudioExtractor.extract_audio(input_video, audio_output_path, audio_sample_rate)
        
        self._print_audio_result(audio_result, audio_output_path)
        
        # Finalize processing
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update metadata with processing step
        step_info = {
            "step": "preprocess",
            "tool": "ffmpeg" if Utils.check_ffmpeg_available() else "opencv",
            "tool_version": "unknown",
            "params": params,
            "runtime_s": round(processing_time, 2),
            "outputs": [f"input/{Path(input_video).name}", "preprocess.json"],
            "status": "success"
        }
        
        if audio_result["success"]:
            step_info["outputs"].append(f"audio/raw.{Config.AUDIO_FORMAT}")
        
        JobManager.update_metadata(job_path, step_info)
        
        # Prepare final results
        results = self._create_success_result(
            job_id, start_time, processing_time, input_video,
            file_size_mb, video_info, audio_result, job_path
        )
        
        JobManager.save_results(job_path, results)
        self._print_completion(processing_time, job_path)
        
        return results
    
    def _print_header(self, job_id: str, input_video: str) -> None:
        """Print processing header information."""
        print("🎬 Starting video preprocessing")
        print(f"📁 Job ID: {job_id}")
        print(f"🎥 Input: {input_video}")
    
    def _print_validation_success(self, video_info: Dict[str, Any], file_size_mb: float) -> None:
        """Print validation success message."""
        resolution = f"{video_info['width']}x{video_info['height']}"
        print(f"✅ Video validated: {video_info['duration_s']}s, {resolution}, {file_size_mb:.1f}MB")
    
    def _print_audio_result(self, audio_result: Dict[str, Any], output_path: Path) -> None:
        """Print audio extraction result."""
        if audio_result["success"]:
            print(f"✅ Audio extracted: {output_path}")
        else:
            print(f"⚠️  Audio extraction failed: {audio_result['error']}")
    
    def _print_completion(self, processing_time: float, job_path: Path) -> None:
        """Print completion summary."""
        print("✅ Preprocessing complete!")
        print(f"📊 Processing time: {processing_time:.2f}s")
        print(f"📁 Job directory: {job_path}")
        print(f"📄 Results saved: {job_path / 'preprocess.json'}")
    
    def _create_error_result(self, error: str, job_id: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            "status": "error",
            "error": error,
            "job_id": job_id
        }
    
    def _create_success_result(
        self,
        job_id: str,
        start_time: datetime,
        processing_time: float,
        input_video: str,
        file_size_mb: float,
        video_info: Dict[str, Any],
        audio_result: Dict[str, Any],
        job_path: Path
    ) -> Dict[str, Any]:
        """Create standardized success result."""
        return {
            "job_id": job_id,
            "status": "success",
            "created_at": start_time.isoformat(),
            "processing_time_s": round(processing_time, 2),
            "input_video": {
                "original_path": str(input_video),
                "archived_path": f"input/{Path(input_video).name}",
                "filename": Path(input_video).name,
                "filesize_mb": round(file_size_mb, 2),
                **video_info
            },
            "audio_extraction": audio_result,
            "job_structure": {
                "job_path": str(job_path),
                "input_dir": "input/",
                "audio_dir": "audio/",
                "logs_dir": "logs/",
                "meta_file": "meta.json"
            }
        }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

class CLIHandler:
    """Command line interface handler for the preprocessing module."""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create and configure argument parser."""
        parser = argparse.ArgumentParser(
            description="Preprocess video for privacy de-identification pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            '--input_video', required=True,
            help='Path to input video file'
        )
        parser.add_argument(
            '--job_id', default=None,
            help='Unique job identifier (auto-generated if not provided)'
        )
        parser.add_argument(
            '--max_duration', type=float, default=Config.DEFAULT_MAX_DURATION,
            help='Maximum allowed video duration in seconds'
        )
        parser.add_argument(
            '--audio_sample_rate', type=int, default=Config.DEFAULT_AUDIO_SAMPLE_RATE,
            help='Audio extraction sample rate'
        )
        parser.add_argument(
            '--jobs_dir', default=Config.DEFAULT_JOBS_DIR,
            help='Base directory for job storage'
        )
        parser.add_argument(
            '--overwrite', action='store_true',
            help='Overwrite existing job directory'
        )
        parser.add_argument(
            '--output_json', action='store_true',
            help='Output results as JSON (for API integration)'
        )
        
        return parser
    
    @staticmethod
    def handle_output(result: Dict[str, Any], output_json: bool) -> int:
        """Handle result output in appropriate format."""
        if output_json:
            print(json.dumps(result, indent=2))
            return 0 if result.get("status") == "success" else 1
        
        # Human-readable output
        if result.get("status") == "success":
            print("\n" + "="*60)
            print("🎉 PREPROCESSING SUCCESSFUL")
            print("="*60)
            print(f"Job ID: {result['job_id']}")
            print(f"Job Path: {result['job_structure']['job_path']}")
            print(f"Processing Time: {result['processing_time_s']}s")
            print(f"Video Duration: {result['input_video']['duration_s']}s")
            print(f"Video Resolution: {result['input_video']['width']}x{result['input_video']['height']}")
            
            if result['audio_extraction']['success']:
                print("Audio: ✅ Extracted to audio/raw.wav")
            else:
                print(f"Audio: ❌ {result['audio_extraction']['error']}")
            
            return 0
        else:
            print(f"\n❌ PREPROCESSING FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1


def main() -> int:
    """Main entry point for video preprocessing."""
    parser = CLIHandler.create_parser()
    args = parser.parse_args()
    
    try:
        # Initialize preprocessor and run pipeline
        preprocessor = VideoPreprocessor()
        result = preprocessor.process(
            input_video=args.input_video,
            job_id=args.job_id,
            max_duration=args.max_duration,
            audio_sample_rate=args.audio_sample_rate,
            jobs_dir=args.jobs_dir,
            overwrite=args.overwrite
        )
        
        return CLIHandler.handle_output(result, args.output_json)
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        error_result = {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "job_id": args.job_id or "unknown"
        }
        
        return CLIHandler.handle_output(error_result, args.output_json)


if __name__ == '__main__':
    sys.exit(main())
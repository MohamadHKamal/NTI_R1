# FDIIV Project Structure

## Directory Structure

```
root/
├─ .gitignore
├─ .fdiivenv/
├─ README.md
├─ requirements.txt
├─ modules/
│  ├─ preprocess/
│  │  └─ preprocess.py        
│  ├─ face_detection/
│  │  └─ detect_and_mask.py   
│  ├─ synthetic_face_generation/
│  │  ├─ generate_dtwin_brushnet.py   
│  │  └─ generate_dtwin_cloud.py      
│  ├─ motion_animation/
│  │  └─ animate.py      
│  ├─ evaluation/
│  │  └─ evaluate.py          
│  └─ deployment/
│     └─ deploy_local.sh      
├─ tools/
│  └─ utils.py                
└─ jobs/
   └─ .gitkeep
```

## Environment Setup

### Create and activate virtual environment

```bash
# Create new virtual environment
python -m venv .fdiivenv

# Activate the environment (Windows)
.fdiivenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Module Descriptions

### preprocess/
- **preprocess.py**: Handles video preprocessing, frame extraction, and audio extraction

### face_detection/
- **detect_and_mask.py**: Face detection, cropping, masking, and landmark detection

### synthetic_face_generation/
- **generate_dtwin_brushnet.py**: BrushNet-based synthetic face generation
- **generate_dtwin_cloud.py**: Cloud-based image editing using Gemini/Qwen APIs

### video_integration/
- **animate_fomm.py**: Animation and motion transfer using FOMM or similar models

### evaluation/
- **evaluate.py**: Evaluation metrics and privacy report generation

### deployment/
- **deploy_local.sh**: Local deployment scripts and helpers

### tools/
- **utils.py**: Utility functions and helpers

### jobs/
- Directory for storing job data and temporary files
# Privacy Video De-Identification Module Calling Convention

## Standardized Direct Script Execution Pattern

All modules in the privacy video de-identification pipeline use **direct script execution** for consistency and simplicity.

### ✅ Standard Pattern

```bash
python modules/{module_name}/{script_name}.py --job jobs/{job_id} [additional_params]
```

### 📋 Module Execution Reference

| Module | Script | Command Pattern |
|--------|--------|-----------------|
| **Preprocess** | `preprocess.py` | `python modules/preprocess/preprocess.py --input_video video.mp4` |
| **Face Detection** | `detect_and_mask.py` | `python modules/face_detection/detect_and_mask.py --input_video video.mp4 --out_dir jobs/{job_id}` |
| **Synthetic Generation** | `generate_dtwin_brushnet.py` | `python modules/synthetic_face_generation/generate_dtwin_brushnet.py --job jobs/{job_id}` |
| **Video Integration** | `animate_fomm.py` | `python modules/video_integration/animate_fomm.py --job jobs/{job_id}` |
| **Evaluation** | `evaluate.py` | `python modules/evaluation/evaluate.py --job jobs/{job_id}` |

### 🔄 Complete Pipeline Example

```bash
# Step 1: Preprocess video (creates job directory)
python modules/preprocess/preprocess.py --input_video video.mp4
# Output: job_20251002_123105_72a91b22

# Step 2: Face detection
python modules/face_detection/detect_and_mask.py \
    --input_video video.mp4 \
    --out_dir jobs/job_20251002_123105_72a91b22

# Step 3: Synthetic face generation (future)
python modules/synthetic_face_generation/generate_dtwin_brushnet.py \
    --job jobs/job_20251002_123105_72a91b22

# Step 4: Video integration (future)  
python modules/video_integration/animate_fomm.py \
    --job jobs/job_20251002_123105_72a91b22

# Step 5: Evaluation (future)
python modules/evaluation/evaluate.py \
    --job jobs/job_20251002_123105_72a91b22
```

### 🎯 Design Principles

1. **No `-m` flag**: Direct script execution only
2. **Consistent parameter names**: `--job` for existing job directories
3. **Job-based workflow**: All modules after preprocess use `--job jobs/{job_id}`
4. **Self-contained**: Each script is independently executable
5. **Clear paths**: Full relative paths from project root

### 📁 Expected Directory Structure

```
modules/
├── preprocess/
│   ├── preprocess.py           # Entry point
│   └── README.md
├── face_detection/
│   ├── detect_and_mask.py      # Main script
│   └── README.md
├── synthetic_face_generation/
│   ├── generate_dtwin_brushnet.py
│   ├── generate_dtwin_cloud.py
│   └── README.md
├── motion_animation/
│   ├── animate.py
│   └── README.md
└── evaluation/
    ├── evaluate.py
    └── README.md
```

### ✅ Benefits of This Convention

- **Simplicity**: No module import complexity
- **Consistency**: Same pattern across all modules
- **Clarity**: Obvious which script is being executed
- **Automation**: Easy to script and automate
- **Documentation**: Clear examples in all READMEs

### 🚫 Avoid These Patterns

```bash
# ❌ Don't use module execution
python -m modules.face_detection.detect_and_mask

# ❌ Don't mix calling conventions
python -m modules.preprocess.preprocess  # inconsistent with others

# ❌ Don't use relative imports without clear path
python detect_and_mask.py  # unclear where script is located
```

### 📝 Implementation Notes

- All scripts should include proper `if __name__ == '__main__':` blocks
- Use absolute imports within modules when needed
- Ensure scripts work when called from project root directory
- Include full path examples in README files
- Test both individual script execution and full pipeline

---
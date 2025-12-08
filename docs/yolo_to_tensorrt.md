# YOLO to TensorRT Conversion Guide

**Author:** Lasantha Kulasooriya
**Date:** December 8, 2024

Convert YOLO models to optimized TensorRT engines for NVIDIA Jetson deployment.

---

## Why TensorRT?

### Performance Benefits

| Metric | PyTorch (FP32) | TensorRT (FP16) | Improvement |
|--------|----------------|-----------------|-------------|
| **Inference Speed** | 40ms | 15ms | **2.7x faster** |
| **Memory Usage** | 800MB | 400MB | **50% less** |
| **Power Efficiency** | Baseline | Optimized | **Better battery life** |

### Key Advantages

1. **Hardware-Specific Optimization**: Engine built for YOUR exact GPU
2. **Kernel Fusion**: Multiple layers combined into single operations
3. **Precision Optimization**: FP16 leverages Jetson Tensor Cores
4. **Memory Efficiency**: Reduced memory footprint for embedded systems
5. **Production Ready**: Stable, tested, NVIDIA-supported

---

## Quick Start

### 1. Install Dependencies

```bash
# Python packages
pip install ultralytics torch tqdm pydantic pydantic-settings pyyaml

# TensorRT (Jetson - usually pre-installed with JetPack)
python3 -c "import tensorrt; print(tensorrt.__version__)"

# If missing:
sudo apt-get install python3-libnvinfer python3-libnvinfer-dev
```

### 2. Configure Your Model

**config/config.yaml:**
```yaml
vision:
  yolo:
    model_path: "models/yolo11n-pose.engine"
    input_size: [640, 640]
    
    tensorrt_engine_settings:
      precision: "FP16"      # FP16, FP32, or INT8
      batch_size: 1          # Usually 1 for real-time
      workspace_size: 2      # GB (adjust for your Jetson)
    
    model_source:
      base_model: "yolo11n-pose"              # Model name
      pt_model_path: "models/yolo11n-pose.pt" # PyTorch model path
```

**Workspace Size Guide:**
| Jetson Model | RAM | Recommended |
|--------------|-----|-------------|
| Nano | 4GB | 1-2 GB |
| Xavier NX | 8GB | 2-4 GB |
| AGX Orin | 32GB+ | 4-8 GB |

### 3. Run Conversion

```bash
# Basic usage (download + convert)
python3 convert_yolo_to_tensorrt.py
```

---

## Command Line Arguments

```bash
python3 convert_yolo_to_tensorrt.py [OPTIONS]
```

### Options

| Argument | Description | Example |
|----------|-------------|---------|
| `--config PATH` | Custom config file path | `--config my_config.yaml` |
| `--force-download` | Re-download PyTorch model | - |
| `--force-convert` | Rebuild TensorRT engine | - |
| `--force-all` | Force both download & convert | - |
| `--download-only` | Download model only, skip conversion | - |
| `--skip-verify` | Skip engine verification | - |
| `--verbose` | Enable debug logging | - |

### Common Usage Examples

```bash
# First time setup
python3 convert_yolo_to_tensorrt.py

# Only download model
python3 convert_yolo_to_tensorrt.py --download-only

# Rebuild engine (after config change)
python3 convert_yolo_to_tensorrt.py --force-convert

# Force complete rebuild
python3 convert_yolo_to_tensorrt.py --force-all

# Custom config
python3 convert_yolo_to_tensorrt.py --config production.yaml

# Verbose output for debugging
python3 convert_yolo_to_tensorrt.py --verbose
```

---

## Conversion Process

```
1. Download
   └─> PyTorch model (.pt) - 2.9MB
       
2. Export to ONNX
   └─> ONNX model (.onnx) - 5.8MB
       
3. Build TensorRT Engine (5-15 minutes)
   ├─> Parse ONNX
   ├─> Optimize layers (kernel fusion, precision conversion)
   ├─> Auto-tune kernels for YOUR GPU
   └─> TensorRT engine (.engine) - 4.2MB
       
4. Verify
   └─> Load and test engine
```

### What Happens During Build?

The build process takes 5-15 minutes because TensorRT:
1. Analyzes all network layers
2. Combines multiple operations (layer fusion)
3. Tests different implementations of each operation
4. Benchmarks performance on YOUR specific hardware
5. Selects optimal kernels for each layer
6. Converts FP32 → FP16 (if configured)

**Result**: Hardware-optimized engine that won't work on different GPUs (by design).

---

## Precision Modes

| Precision | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| **FP32** | 1x | 100% | Development/testing |
| **FP16** ⭐ | 2-3x | ~99.5% | **Production (recommended)** |
| **INT8** | 4-5x | ~97% | Edge devices (needs calibration) |

**Recommendation**: Use FP16 for Jetson - best speed/accuracy tradeoff.

---

## Supported Models

### Format
```
yolo[v]<version><size>-<task>
```

### Examples
- `yolo11n-pose` - YOLO11 nano pose estimation
- `yolov8s-detect` - YOLOv8 small object detection
- `yolo10m-seg` - YOLO10 medium segmentation

### Versions
- YOLOv8 (`yolov8` or `yolo8`)
- YOLOv9 (`yolov9` or `yolo9`)
- YOLOv10 (`yolov10` or `yolo10`)
- YOLO11 (`yolo11` or `yolov11`)

### Sizes
| Size | Speed | Accuracy | Params |
|------|-------|----------|--------|
| `n` | Fastest | Good | 2.9M |
| `s` | Fast | Better | 9.4M |
| `m` | Medium | Best | 20M |
| `l` | Slow | Higher | 25M |
| `x` | Slowest | Highest | 68M |

### Tasks
- `pose` - Keypoint detection
- `detect` - Object detection
- `seg` - Instance segmentation
- `obb` - Oriented bounding boxes
- `cls` - Classification

---

## Using the Engine

```python
from ultralytics import YOLO
from config.settings import get_settings

# Load settings
settings = get_settings()

# Load TensorRT engine
model = YOLO(settings.vision.yolo.model_path)

# Run inference
results = model(image)

# Process results
for result in results:
    keypoints = result.keypoints  # Pose keypoints
    boxes = result.boxes          # Bounding boxes
    confidences = result.probs    # Confidence scores
```

---

## Troubleshooting

### Out of Memory During Build

**Error**: `CUDA out of memory`

**Solution**: Reduce workspace size
```yaml
tensorrt_engine_settings:
  workspace_size: 1  # Reduce from 2
```

### TensorRT Not Found

**Error**: `ModuleNotFoundError: No module named 'tensorrt'`

**Solution**:
```bash
# Jetson
sudo apt-get install python3-libnvinfer python3-libnvinfer-dev

# x86 Linux
pip install tensorrt
```

### Build Takes Too Long

**Expected**: 5-15 minutes for first build
- Nano models: 5-8 minutes
- Small models: 8-12 minutes
- Medium models: 12-20 minutes

This is normal! TensorRT is optimizing for your hardware.

### Engine Verification Failed

**Solution**: Rebuild engine
```bash
python3 scripts/yolo_to_tensorrt.py --force-convert
```

---

## File Structure

After conversion:
```
project/
├── config/
│   ├── config.yaml
│   └── settings.py
├── models/
│   ├── yolo11n-pose.pt       # PyTorch model
│   ├── yolo11n-pose.onnx     # ONNX intermediate (optional to keep)
│   └── yolo11n-pose.engine   # TensorRT engine (USE THIS)
└── scripts
    └── yolo_to_tensorrt.py
```

---

## Additional Resources

### Official Documentation
- **TensorRT**: https://docs.nvidia.com/deeplearning/tensorrt/
- **Ultralytics YOLO**: https://docs.ultralytics.com
- **NVIDIA Jetson**: https://developer.nvidia.com/embedded/jetson

### Key Concepts
- **TensorRT Developer Guide**: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
- **TensorRT Python API**: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/
- **ONNX Format**: https://onnx.ai/

### Performance Optimization
- **TensorRT Best Practices**: https://docs.nvidia.com/deeplearning/tensorrt/best-practices/
- **Jetson Performance Tuning**: https://developer.nvidia.com/embedded/jetson-performance

### Related Tools
- **NVIDIA Nsight Systems**: Profile GPU performance
- **trtexec**: Command-line TensorRT tool
- **Polygraphy**: TensorRT debugging tool

---

## FAQ

**Q: Do I need to rebuild the engine for different Jetson devices?**  
A: Yes. Each engine is hardware-specific.

**Q: Can I use the same engine on different input sizes?**  
A: No. Engine is built for specific input size. Change `input_size` in config and rebuild.

**Q: Why is FP16 recommended over INT8?**  
A: FP16 offers better speed/accuracy tradeoff. INT8 requires calibration and loses more accuracy.

**Q: Can I convert custom-trained YOLO models?**  
A: Yes! Export your model to `.pt` format and use the same process.

**Q: How much faster is TensorRT compared to PyTorch?**  
A: Typically 2-3x faster on Jetson with FP16, up to 5x with INT8.

**Q: Does the engine work on other GPUs?**  
A: No. Each engine is optimized for the specific GPU it was built on.

---

## Summary

✅ **2-3x faster** inference on Jetson  
✅ **50% less memory** usage  
✅ **Hardware-optimized** for your specific GPU  
✅ **Production-ready** with NVIDIA support  
✅ **Simple one-command** conversion  

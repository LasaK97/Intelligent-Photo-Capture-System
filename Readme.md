# Intelligent Photo Capture System 
## Complete Technical Design Document

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Hardware Architecture](#hardware-architecture)
3. [Key Technical Decisions](#key-technical-decisions)
4. [Scene Classification Logic](#scene-classification-logic)
5. [Zoom Calculation Mathematics](#zoom-calculation-mathematics)
6. [Gimbal Control Specifications](#gimbal-control-specifications)
7. [State Machine Workflow](#state-machine-workflow)
8. [Focus Calibration System](#focus-calibration-system)
9. [Depth Processing Pipeline](#depth-processing-pipeline)
10. [ROS2 Topic Architecture](#ros2-topic-architecture)
11. [Configuration Files](#configuration-files)
12. [Implementation Roadmap](#implementation-roadmap)
13. [DeepStream Analysis](#deepstream-analysis)

---

## System Overview

An autonomous robotic photography system combining computer vision, depth sensing, and voice guidance to capture professional-quality photos without human intervention. Acts as an **AI photographer** by:

- Detecting subjects
- Guiding positioning through natural language
- Automatically adjusting camera settings
- Capturing when composition conditions are met

### Core Capabilities

- **Real-time multi-person detection** using YOLOv8s on TensorRT
- **Depth-based positioning** via Intel RealSense D455
- **Voice-guided subject positioning** ("Please take 2 steps back")
- **Automatic scene classification** (portrait, couple, small/medium/large groups)
- **Smart gimbal control** via DJI RS 3 Pro with ros2_control integration
- **DIY autofocus system** using depth data → DJI Focus Motor commands

---

## Hardware Architecture

| Component | Specification |
|-----------|---------------|
| **Platform** | NVIDIA Jetson AGX Orin |
| **Camera** | Canon R6 Mark II + RF 24-105mm f/2.8L IS USM Z lens |
| **Gimbal** | DJI RS 3 Pro (CAN bus control via can3) |
| **Focus Motor** | DJI Focus Motor FM03 (mechanical motor, NOT LiDAR-based) |
| **Depth Sensor** | Intel RealSense D455 (1280x720 @ 30fps depth + 1920x1080 RGB) |
| **Robot Base** | Manriix 4-wheel steering platform (0.53m above ground) |

### Mounting Heights

- **Canon Camera:** 1.68m from ground
- **RealSense:** 1.40m from ground
- **Vertical Offset:** 0.28m

---

## Key Technical Decisions

### 1. Focus Motor Interface
- **Topic:** `/set_focus_position` (std_msgs/Int32)
- Separate ROS2 node publishes motor position (0-4095 range)
- Calculated from depth + calibration lookup table

### 2. Zoom Control
- **Manual:** Operator sets lens to ~50mm before session
- System assumes fixed focal length
- Uses gimbal pan/tilt for framing (NOT motorized zoom)

### 3. Gimbal Control Method
**Dual Strategy:**
- `JointTrajectoryController` (action server, smooth trajectories) for large movements
- Direct commands (`/gimbal_controller/commands` topic) for fine-tuning

### 4. Detection Model
- **Model:** YOLOv8s TensorRT FP16
- **Performance:** 100-120 FPS @ 640x640 on Orin
- **Balance:** Speed/accuracy optimized

### 5. Depth Processing
- Uses aligned depth from RealSense (`/camera/aligned_depth_to_color/image_raw`)
- Extracts center 40% of person bounding box
- Applies median filtering
- Validates with depth variance checks

---

## Scene Classification Logic

| Scene Type | People | Target Distance | Focal Length | Target Spread |
|------------|--------|-----------------|--------------|---------------|
| **Portrait** | 1 | 3.0-4.0m | 70-85mm equiv | N/A |
| **Couple** | 2 | 2.5-3.5m | 50-70mm | <0.8m (intimate) |
| **Small Group** | 3-4 | 4.0-5.5m | 40-55mm | 1.5-2.5m |
| **Medium Group** | 5-7 | 5.0-6.5m | 30-45mm | 2.5-4.0m |
| **Large Group** | 8-10 | 6.0-8.0m | 24-35mm | 3.5-6.0m |

---

## Zoom Calculation Mathematics

### Horizontal FOV Formula

```
HFOV = 2 × arctan(sensor_width / (2 × focal_length))
```

**For Canon R6 Mark II full-frame (36mm × 24mm sensor) at 50mm lens:**

```
HFOV = 2 × arctan(36mm / (2 × 50mm))
     = 2 × arctan(0.36)
     = 39.6°
```

### Forward Calculation (Given focal length and distance, find scene width)

```
Scene_Width = 2 × D × tan(HFOV / 2)
```

**Example:** At 4.5m with 50mm lens:
```
Scene_Width = 2 × 4.5m × tan(19.8°) = 3.24m
```

### Reverse Calculation (Given group width W and distance D, find required focal length)

```python
# Step 1: Calculate required HFOV
Required_HFOV = 2 × arctan(W / (2 × D))

# Step 2: Add composition padding
W_with_padding = W × (1 + 2 × padding%)
Required_HFOV_padded = 2 × arctan(W_with_padding / (2 × D))

# Step 3: Calculate required focal length
f_required = sensor_width / (2 × tan(Required_HFOV_padded / 2))
```

**Example:** 1.8m group at 4.5m with 10% padding:
```
W_with_padding = 1.8m × 1.20 = 2.16m
Required_HFOV = 2 × arctan(2.16 / 9.0) = 27°
f_required = 36mm / (2 × tan(13.5°)) = 75mm
```

### System Logic (Since Zoom is Manual ~50mm fixed)

1. **Check if subjects fit in current FOV**
2. **If group too wide** → Guide: "Please move closer together"
3. **If group too narrow** → Ask people to move back (increases Captured_Width)
4. **Fine-tune framing** via gimbal pan/tilt

---

## Gimbal Control Specifications

### Joint Configuration (from URDF)

| Joint | Function | Range | Zero Position |
|-------|----------|-------|---------------|
| joint_5 | Yaw/Pan | -180° to +180° (-3.14 to 3.14 rad) | Facing forward |
| joint_6 | Pitch/Tilt | -45° to +45° (-0.784 to 0.784 rad) | Level/horizontal |
| joint_7 | Roll | -45° to +45° | NOT USED (kept at 0°) |

### Control Parameters

```yaml
max_angular_velocity: 0.349 rad/s  # ~20°/s from DJI specs
settle_time: 1.0s                   # Seconds after movement for stabilization
position_tolerance: 0.0087 rad      # ~0.5°
horizontal_deadzone: 0.0087 rad
vertical_deadzone: 0.0087 rad
```

### Centering Calculation

```python
# Pan error calculation
pan_error = arctan(offset_x / distance)
# where offset_x = person.x_world (positive = right, negative = left)

# Tilt error calculation  
tilt_error = arctan(offset_y / distance)
# where offset_y adjusted for camera height
```

### Height Adjustment

- **Target face position:** 0.4 (40% from top of frame, faces in upper-middle third)
- Calculates average face height
- Compares to `frame_height × 0.4`
- Adjusts gimbal tilt if needed
- **For children:** Estimates average person height, adds headroom (15-20%), checks if fits

### Smoothing

Exponential moving average with factor **0.4** to avoid jittery movements

---

## State Machine Workflow

```
IDLE → INIT → INVITE → DETECT_SCENE → ANALYZE_POSITION → GUIDE
                                                          ↓
COMPLETE ← CAPTURE ← COUNTDOWN ← VERIFY ← ADJUST_CAMERA ←─┘
```

### State Descriptions

| State | Duration | Description |
|-------|----------|-------------|
| **INIT** | 2s | Initialize gimbal to zero position (pan=0°, tilt=0°), start frame capture, reset variables |
| **INVITE** | 10s timeout | Output "Hello! Please come in front of me", wait for person detection >1 second |
| **DETECT_SCENE** | 0.5s | Classify scene type, determine target distance/spread, calculate initial camera settings |
| **ANALYZE_POSITION** | - | Calculate positioning errors, determine guidance priority |
| **GUIDE** | max 10 iterations, 60s total | Output positioning guidance, wait 3s for movement, re-analyze |
| **ADJUST_CAMERA** | - | Calculate gimbal angles, publish focus motor position, wait 1s for stabilization |
| **VERIFY** | 2s | Re-run detection and position check, verify all conditions met |
| **COUNTDOWN** | 3.5s | "Perfect! Please smile" (1.5s), "Three" (1s), "Two" (1s), "One" (1s), pause (0.5s) |
| **CAPTURE** | - | Publish 'y' to `/capture_image_topic`, Canon captures RAW+JPEG |
| **COMPLETE** | - | "All done! Thank you!", cleanup, ready for next activation |

### Guidance Priority Order

1. `CRITICAL_DISTANCE` (priority: 10)
2. `CRITICAL_POSITION` (priority: 8)
3. `MODERATE_SPREAD` (priority: 6)
4. `MINOR_FINE_TUNE` (priority: 3)

### Verify Conditions

- Distance error < 0.3m
- Horizontal error < 0.2m
- Spread acceptable
- Faces detected and oriented
- Gimbal stabilized
- Focus locked
- Lighting check passed

---

## Focus Calibration System

### Calibration File Format

```yaml
# focus_calibration.yaml
focal_length_24mm:
  0.5: 150
  1.0: 520
  2.0: 1100
  3.0: 1680
  5.0: 2420
  10.0: 3350

focal_length_50mm:
  0.5: 100
  1.0: 450
  2.0: 1150
  3.0: 1573
  5.0: 2420
  10.0: 3400
```

### Calibration Process

1. Mount camera + lens on gimbal
2. Place target at known distances (0.5m, 1m, 2m, 3m, 5m, 10m)
3. Manually focus camera to sharp focus
4. Record motor position for each distance
5. Build lookup table

### Focus Distance Calculation

```python
# Example: Couple at 3.0m average depth
optimal_focus_distance = 3.0  # meters
current_focal_length = 50     # mm (manually set)

# Load calibration
calibration_data = focus_calibration['focal_length_50mm']

# Direct lookup or interpolation
if 3.0 in calibration_data:
    motor_position = calibration_data[3.0]  # = 1573
else:
    # Linear interpolation between surrounding points
    # Example: 3.0m between 2.0m (1150) and 5.0m (2420)
    ratio = (3.0 - 2.0) / (5.0 - 2.0)  # = 0.333
    motor_position = 1150 + 0.333 × (2420 - 1150)  # = 1573

# Publish command
# ros2 topic pub --once /set_focus_position std_msgs/Int32 "{data: 1573}"
```

---

## Depth Processing Pipeline

### Threading Architecture

| Thread | Function | Rate |
|--------|----------|------|
| Thread 1 | Frame Acquisition (Canon RGB via FrameClient TCP socket port 8089) | 30 FPS |
| Thread 2 | Depth Reception (RealSense D455 via pyrealsense2 pipeline) | 30 FPS |
| Thread 3 | Vision Processing (YOLOv8s TensorRT) | 10-15 FPS |

### Person-to-Depth Mapping

```python
# For each detected person:

# 1. Extract bounding box from Canon frame (1920x1080)
bbox = (x1, y1, x2, y2)

# 2. Calculate bbox center
bbox_center_x = (x1 + x2) / 2
bbox_center_y = (y1 + y2) / 2

# 3. Map to aligned depth frame
depth_x = int(bbox_center_x × (depth_width / canon_width))
depth_y = int(bbox_center_y × (depth_height / canon_height))

# 4. Extract depth region (40% center of bbox for robustness)
depth_region_width = int((x2 - x1) × 0.4)
depth_region_height = int((y2 - y1) × 0.4)

# 5. Extract depth values
depth_roi = aligned_depth[
    depth_y - depth_region_height//2 : depth_y + depth_region_height//2,
    depth_x - depth_region_width//2 : depth_x + depth_region_width//2
]

# 6. Filter outliers
valid_depths = depth_roi[depth_roi > 0]

# 7. Calculate robust depth metrics
median_depth = np.median(valid_depths)
depth_variance = np.var(valid_depths)

# 8. Convert to meters
person.depth_m = median_depth / 1000.0  # RealSense gives mm
```

### 3D Position Calculation

```python
# OpenCV convention: X: Right, Y: Down, Z: Forward/depth

# Using RealSense intrinsics from camera_info topic
intrinsics = rs_depth_frame.profile.as_video_stream_profile().intrinsics
point_3d = rs2.rs2_deproject_pixel_to_point(
    intrinsics, 
    [depth_x, depth_y], 
    median_depth
)

person.x_world = point_3d[0] / 1000.0  # meters, horizontal right+
person.y_world = point_3d[1] / 1000.0  # meters, vertical down+
person.z_world = point_3d[2] / 1000.0  # meters, depth forward+

# Adjust for camera height (convert to ground-relative coordinates)
person.height_from_ground = camera_height - person.y_world
# Example: camera_height = 1.68m, person.y_world = -0.05m
# height_from_ground = 1.68 - (-0.05) = 1.73m ≈ 173cm ✓
```

### Depth Reliability Check

```python
def check_depth_reliability(depth_roi):
    valid_pixels = np.count_nonzero(depth_roi > 0)
    total_pixels = depth_roi.size
    
    # Check conditions
    is_valid = (
        valid_pixels > 0 and
        np.max(depth_roi) < np.inf and
        np.var(depth_roi[depth_roi > 0]) < 0.3 and  # <0.3m variance
        (valid_pixels / total_pixels) > 0.7  # >70% confidence
    )
    
    return is_valid
```

---

## ROS2 Topic Architecture

### Input Topics (Subscribe)

| Topic | Type | Description |
|-------|------|-------------|
| `/va_photo_capture` | std_msgs/String | Activation trigger ('y' to start) |
| `/camera/aligned_depth_to_color/image_raw` | sensor_msgs/Image | RealSense D455 depth (16UC1, mm) |
| `/camera/aligned_depth_to_color/camera_info` | sensor_msgs/CameraInfo | Intrinsics for 3D calculation |
| `/joint_states` | sensor_msgs/JointState | Current gimbal angles |
| **Canon Frame** | TCP socket (port 8089) | High-quality RGB via FrameClient |

### Output Topics (Publish)

| Topic | Type | Description |
|-------|------|-------------|
| `/gimbal_controller/follow_joint_trajectory` | FollowJointTrajectory (action) | Smooth gimbal movements |
| `/gimbal_controller/commands` | std_msgs/Float64MultiArray | Direct gimbal commands [yaw, pitch, roll] |
| `/set_focus_position` | std_msgs/Int32 | Focus motor position (0-4095) |
| `/capture_image_topic` | std_msgs/String | Trigger Canon capture ('y') |
| `/tts_text_output` | std_msgs/String | Guidance phrases for voice agent |
| `/photo_capture/person_detections` | Detection2DArray | Debug: detected people |
| `/photo_capture/state` | std_msgs/String | Current state machine state |
| `/photo_capture/composition_frame` | sensor_msgs/Image | Debug: annotated frame |

---

## Configuration Files

### Main Config: `photo_capture.yaml`

```yaml
system:
  photos_per_session: 3
  max_positioning_time: 60.0  # seconds
  guidance_wait_time: 3.0     # seconds
  timeout_behavior: "wait_for_reactivation"
```

### Gimbal Config: `gimbal.yaml`

```yaml
gimbal:
  control_method: "hybrid"  # "action", "direct", or "hybrid"
  joint_names:
    - "joint_5"  # yaw/pan
    - "joint_6"  # pitch/tilt
    - "joint_7"  # roll (unused)
  
  limits:
    pan_min: -3.14
    pan_max: 3.14
    tilt_min: -0.784
    tilt_max: 0.784
  
  action_server: "/gimbal_controller/follow_joint_trajectory"
  command_topic: "/gimbal_controller/commands"
  state_topic: "/joint_states"
  
  init_position: [0.0, 0.0, 0.0]
  smoothing_factor: 0.4
  max_angular_velocity: 0.349  # rad/s
  settle_time: 1.0
  
  deadzone:
    horizontal: 0.0087  # rad (~0.5°)
    vertical: 0.0087
```

### Hardware Config: `hardware.yaml`

```yaml
hardware:
  canon_camera_height: 1.68  # meters
  realsense_height: 1.40     # meters
  vertical_offset: 0.28      # meters
  
  frame_client:
    host: "localhost"
    port: 8089
  
  realsense:
    depth_topic: "/camera/aligned_depth_to_color/image_raw"
    camera_info_topic: "/camera/aligned_depth_to_color/camera_info"
    depth_range_min: 0.3   # meters
    depth_range_max: 10.0  # meters
    
    filters:
      spatial: true
      temporal: true
      hole_filling: true
```

### Camera Config: `camera.yaml`

```yaml
camera:
  model: "Canon R6 Mark II"
  sensor:
    width_mm: 36.0
    height_mm: 24.0
  
  lens:
    model: "RF 24-105mm f/2.8L IS USM Z"
    focal_length_min: 24
    focal_length_max: 105
    aperture: "f/2.8"
  
  default_focal_length: 50  # mm (manual zoom)
  # Note: Zoom is manual, not motorized
```

### Detection Config: `detection.yaml`

```yaml
detection:
  model: "yolov8s"
  model_path: "/models/yolov8s.engine"
  input_size: 640
  confidence_threshold: 0.6
  nms_threshold: 0.45
  max_persons: 10
  use_tensorrt: true
  device: "cuda:0"
  processing_rate: 10.0  # Hz
  frame_skip: 3
```

### Composition Config: `composition.yaml`

```yaml
composition:
  portrait:
    person_count: 1
    target_distance: 3.5
    optimal_distance: [3.0, 4.0]
    target_focal_length: 75
    padding:
      horizontal: 0.15
      vertical: 0.20
    headroom: 0.15
  
  couple:
    person_count: 2
    target_distance: 3.0
    optimal_distance: [2.5, 3.5]
    target_focal_length: 60
    target_spread: 0.8
    padding:
      horizontal: 0.12
      vertical: 0.18
  
  small_group:
    person_count: [3, 4]
    target_distance: 4.75
    optimal_distance: [4.0, 5.5]
    target_focal_length: 47
    target_spread: [1.5, 2.5]
    padding:
      horizontal: 0.10
      vertical: 0.15
  
  medium_group:
    person_count: [5, 7]
    target_distance: 5.75
    optimal_distance: [5.0, 6.5]
    target_focal_length: 37
    target_spread: [2.5, 4.0]
    padding:
      horizontal: 0.10
      vertical: 0.12
  
  large_group:
    person_count: [8, 10]
    target_distance: 7.0
    optimal_distance: [6.0, 8.0]
    target_focal_length: 30
    target_spread: [3.5, 6.0]
    padding:
      horizontal: 0.08
      vertical: 0.10
```

### Positioning Config: `positioning.yaml`

```yaml
positioning:
  tolerances:
    distance: 0.3      # meters
    horizontal: 0.2    # meters
    spread: 0.3        # meters
  
  step_size: 0.4       # meters per step
  max_iterations: 10
  
  guidance_priority:
    critical_distance: 10
    critical_horizontal: 8
    moderate_spread: 6
    minor_adjustment: 3
```

### Guidance Config: `guidance.yaml`

```yaml
guidance:
  language: "en"
  
  phrases:
    init: "Initializing camera system..."
    analyzing: "Let me take a look..."
    ready: "Perfect! Please smile"
    countdown: ["Three", "Two", "One"]
    complete: "All done! Thank you, your photos look wonderful!"
    timeout: "I'm having trouble getting the perfect shot. Let's try again."
    next_photo: "Great! Let's take another one."
  
  templates:
    move_back: "Please take {steps} step(s) back"
    move_forward: "Please take {steps} step(s) forward"
    move_left: "Please move a little to your left"
    move_right: "Please move a little to your right"
    spread_out: "Please spread out a little more"
    move_closer: "Please move closer together"
    person_specific: "{person}, please move {direction}"
```

### State Machine Config: `state_machine.yaml`

```yaml
state_machine:
  states:
    - IDLE
    - INIT
    - INVITE
    - DETECT_SCENE
    - ANALYZE_POSITION
    - GUIDE
    - ADJUST_CAMERA
    - VERIFY
    - COUNTDOWN
    - CAPTURE
    - COMPLETE
    - TIMEOUT
    - ERROR
  
  timeouts:
    invite: 10.0           # seconds
    guidance_cycle: 3.0    # seconds
    total_positioning: 60.0  # seconds
    verify: 2.0            # seconds
  
  transitions:
    min_detection_duration: 1.0  # seconds
```

---

## Implementation Roadmap

### Phase 1: Calibration (1-2 days)

1. Focus motor calibration at different focal lengths
2. RealSense-Canon extrinsic calibration
3. Test focus accuracy at various distances

### Phase 2: Core System (1 week)

1. ROS2 Humble node structure
2. Person detection + depth fusion
3. Focus distance calculation + motor control
4. Basic state machine (INIT → DETECT → FOCUS → CAPTURE)

### Phase 3: Smart Positioning (1 week)

1. Guidance message generation
2. Multi-person coordination
3. Composition optimization
4. Gimbal control integration

### Phase 4: Production Features (1 week)

1. Error handling and recovery
2. Voice agent integration
3. Multi-photo sessions
4. Performance optimization

---

## DeepStream Analysis

### Current Setup

- Python script
- Load TensorRT model
- Run inference (~100 lines)

### DeepStream Setup (Alternative)

- GStreamer pipeline configuration
- DeepStream config files (5+ files)
- Custom plugins if needed
- ROS2 bridge for integration
- ~500+ lines + config complexity

### Comparison

| Aspect | Current (TensorRT) | DeepStream |
|--------|-------------------|------------|
| Speed Gain | Baseline | ~30-60% faster |
| Complexity | Low | High |
| Person Tracking | Manual | Built-in |
| Development Time | Fast | 2-3 days extra |

### When to Use DeepStream

**DON'T Use DeepStream IF:**
- Current YOLOv8 TensorRT inference is fast enough (AGX Orin has plenty of power)
- Processing 10-15 FPS (not 30+ FPS)
- Single camera
- Simple detection task
- Want faster development time

**DO Use DeepStream IF:**
- Planning to add multiple cameras (4+ streams)
- Need video recording with analytics overlay
- Want built-in person tracking/re-identification
- Need absolute minimum latency (<30ms)
- Building production-grade surveillance system

### Verdict

**NOT worth it for this project** unless you plan to add 3+ more cameras or need video recording.

---

## Practical Example: Family of 4

### Setup

- **Lens:** 50mm (manually set)
- **HFOV:** 39.6°
- **Family width:** 2.0m (spread out)
- **Distance:** 4.0m
- **Target padding:** 10%

### Step 1: Calculate What Camera Sees

```
Captured_Width = 2 × 4.0m × tan(39.6° / 2)
               = 2 × 4.0m × 0.36
               = 2.88 meters
```

### Step 2: Calculate Required Width with Padding

```
Required_Width = 2.0m × (1 + 2 × 0.10)
               = 2.0m × 1.20
               = 2.4 meters
```

### Step 3: Check Fit

```
2.4m < 2.88m ✓ FITS!
```

### Step 4: Calculate Positioning Accuracy

```
Extra_Space = 2.88m - 2.4m = 0.48m (24cm on each side)
Fit_Quality = 2.4 / 2.88 = 83% (good framing)
```

### Decision

✓ **Framing is acceptable, proceed with capture**

---

## What If Group Doesn't Fit?

### Scenario

- 3.5m group at 4.0m distance

```
Captured_Width = 2 × 4.0m × tan(19.8°) = 2.88m
Required_Width = 3.5m × 1.20 = 4.2m
4.2m > 2.88m ❌ DOESN'T FIT!
```

### Solutions

**Option 1: Move People BACK**
```
New_Distance = 4.2m / (2 × 0.36) = 5.8m
Guidance: "Everyone, please take 4 steps back"
(1.8m difference ÷ 0.4m per step = 4.5 steps)
```

**Option 2: Move People CLOSER Together**
```
Target_Width = 2.88m / 1.2 = 2.4m
Reduction_Needed = 3.5m - 2.4m = 1.1m
Guidance: "Please move closer together"
(Each person moves ~0.27m toward center)
```

**Option 3: Suggest Zoom Change (Manual Intervention)**
```
Required_Focal_Length ≈ 32mm
Guidance: "Please set lens to wide angle (~35mm)"
```

---

## Key Takeaways

1. **DeepStream:** Not necessary for single-camera photo capture, but excellent if expanding to multi-camera system

2. **Zoom Calculation:** System doesn't control zoom, but uses geometry to:
   - Check if subjects fit in current FOV
   - Guide subjects to optimal positions
   - Fine-tune framing with gimbal movements

3. **Configuration:** Dual gimbal control method + separate composition.yaml gives maximum flexibility

4. **The Math:** It all comes down to trigonometry—`tan()` converts angles ↔ distances

---

## File Structure

```
mapcs_ws/
├── src/
│   └── mapcs/
│       ├── mapcs/
│       │   ├── __init__.py
│       │   ├── photo_capture_node.py
│       │   ├── detection/
│       │   │   ├── __init__.py
│       │   │   ├── yolo_detector.py
│       │   │   └── person_tracker.py
│       │   ├── depth/
│       │   │   ├── __init__.py
│       │   │   └── depth_processor.py
│       │   ├── gimbal/
│       │   │   ├── __init__.py
│       │   │   └── gimbal_controller.py
│       │   ├── focus/
│       │   │   ├── __init__.py
│       │   │   └── focus_controller.py
│       │   ├── composition/
│       │   │   ├── __init__.py
│       │   │   └── scene_analyzer.py
│       │   ├── guidance/
│       │   │   ├── __init__.py
│       │   │   └── guidance_generator.py
│       │   ├── state_machine/
│       │   │   ├── __init__.py
│       │   │   └── capture_state_machine.py
│       │   └── utils/
│       │       ├── __init__.py
│       │       ├── frame_client.py
│       │       └── config_loader.py
│       ├── config/
│       │   ├── photo_capture.yaml
│       │   ├── gimbal.yaml
│       │   ├── hardware.yaml
│       │   ├── camera.yaml
│       │   ├── detection.yaml
│       │   ├── composition.yaml
│       │   ├── positioning.yaml
│       │   ├── guidance.yaml
│       │   ├── state_machine.yaml
│       │   ├── focus_calibration.yaml
│       │   └── ros2.yaml
│       ├── launch/
│       │   └── mapcs.launch.py
│       ├── resource/
│       │   └── mapcs
│       ├── test/
│       │   ├── test_detection.py
│       │   ├── test_depth.py
│       │   └── test_gimbal.py
│       ├── package.xml
│       ├── setup.py
│       └── setup.cfg
└── models/
    └── yolov8s.engine
```

---

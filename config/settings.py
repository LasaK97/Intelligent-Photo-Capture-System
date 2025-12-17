import os
from pathlib import Path
from typing import List, Tuple, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings
import yaml


def get_config_path(config_filename: str, env_var: str) -> str:
    """find configs"""
    if env_var in os.environ:
        env_path = Path(os.environ[env_var])
        if env_path.exists():
            return str(env_path)

    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_share = Path(get_package_share_directory('manriix_photo_va'))
        ros_config_path = pkg_share / 'config' / config_filename
        if ros_config_path.exists():
            return str(ros_config_path)
    except (ImportError, Exception):
        # ament_index_python not available or package not found
        pass

    cwd_config = Path.cwd() / 'config' / config_filename
    if cwd_config.exists():
        return str(cwd_config)

    file_dir = Path(__file__).parent
    dev_config_path = file_dir / config_filename
    if dev_config_path.exists():
        return str(dev_config_path)

    # Priority 5: Check parent directory (in case running from subdirectory)
    parent_config = file_dir.parent / 'config' / config_filename
    if parent_config.exists():
        return str(parent_config)

    # If still not found, raise helpful error
    searched_paths = [
        f"Environment variable: {env_var}",
        "ROS2 package share directory",
        f"Current working directory: {cwd_config}",
        f"Development directory: {dev_config_path}",
        f"Parent config directory: {parent_config}"
    ]

    raise FileNotFoundError(
        f"Config file '{config_filename}' not found. Searched:\n" +
        "\n".join(f"  - {p}" for p in searched_paths)
    )


def get_model_path(model_filename: str, env_var: str = 'MODELS_PATH') -> str:
    """Find model files using same priority as get_config_path"""

    if env_var in os.environ:
        env_path = Path(os.environ[env_var])
        if env_path.is_dir():
            model_path = env_path / model_filename
            if model_path.exists():
                return str(model_path)
        elif env_path.exists() and env_path.name == model_filename:
            return str(env_path)

    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_share = Path(get_package_share_directory('manriix_photo_va'))
        ros_model_path = pkg_share / 'models' / model_filename
        if ros_model_path.exists():
            return str(ros_model_path)
    except (ImportError, Exception):
        pass

    cwd_model = Path.cwd() / 'models' / model_filename
    if cwd_model.exists():
        return str(cwd_model)

    file_dir = Path(__file__).parent
    dev_model_path = file_dir.parent / 'models' / model_filename
    if dev_model_path.exists():
        return str(dev_model_path)

    parent_model = file_dir.parent.parent / 'models' / model_filename
    if parent_model.exists():
        return str(parent_model)

    searched_paths = [
        f"Environment variable: {env_var}",
        "ROS2 package share directory",
        f"Current working directory: {cwd_model}",
        f"Development directory: {dev_model_path}",
        f"Parent directory: {parent_model}"
    ]

    raise FileNotFoundError(
        f"Model file '{model_filename}' not found. Searched:\n" +
        "\n".join(f"  - {p}" for p in searched_paths)
    )
# =======================================================================
#                         config.yaml validations
# =======================================================================

class ModelSource(BaseModel):
    """Yolo download and conversion settings"""
    base_model: str = Field(..., description="Base YOLO model name")
    download_url: Optional[str] = Field(default=None, description="Custom download URL (Auto generated if None")
    pt_model_path: str = Field(... , description="Path to the pytorch model file")
    model_urls: Dict[str, str] = Field(
        default={
            "8": "https://github.com/ultralytics/assets/releases/download/v8.3.0/",
            "v8": "https://github.com/ultralytics/assets/releases/download/v8.3.0/",
            "9": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
            "v9": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
            "10": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
            "v10": "https://github.com/ultralytics/assets/releases/download/v8.2.0/",
            "11": "https://github.com/ultralytics/assets/releases/download/v8.3.0/",
            "v11": "https://github.com/ultralytics/assets/releases/download/v8.3.0/",
        },
        description="YOLO model download URLs by version"
    )

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, v: str) -> str:
        """Validate base model"""
        import re
        pattern = r'^yolo(v)?(\d{1,2})(n|s|m|l|x)-(pose|seg|detect|obb|cls)$'
        if not re.match(pattern, v.lower()):
            raise ValueError(f"Invalid base YOLO model name: {v}")
        return v.lower()


class TensorRTEngineSettings(BaseModel):
    """TensorRT engine settings"""
    precision: Literal["FP16", "FP32", "INT8"] = Field(default="FP16", description="TensorRT precision mode")
    batch_size: int = Field(default=1, ge=1, le=32,  description="Batch size for inference")
    workspace_size: int = Field(default=4, ge=1, le=16, description="TensorRT workspace size in GB")


class YOLOConfig(BaseModel):
    """Yolo-pose configs"""

    model_path: str = Field(..., description="Path to TensorRT model file")
    confidence_threshold: float = Field(0.6, ge=0.1, le=1.0, description=" YOLO Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.1, le=1.0, description=" YOLO IoU threshold")
    max_detections: int = Field(8, ge=1, le=20, description="Max number of detections")
    input_size: Tuple[int, int] = Field((640, 640), description="Input image size")
    device: int = Field(0, ge=0, le=1, description="Device index [GPU/ CPU]")
    tensorrt_engine_settings: TensorRTEngineSettings = Field(default_factory=TensorRTEngineSettings, description="TensorRT engine settings")
    model_source: ModelSource = Field(..., description="Model source download and conversion settings")

    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        if not Path(v).suffix == '.engine':
            raise ValueError(f'Invalid model-[Model must be a TensorRT engine]: {v}')
        try:
            filename = Path(v).name
            return get_model_path(filename)
        except FileNotFoundError:
            import warnings
            warnings.warn(f"Model file not found: {v}.")
            return v

    @model_validator(mode='after')
    def resolve_pt_model_path(self) -> 'YOLOConfig':
        """Resolve PT model path in model_source"""
        if hasattr(self.model_source, 'pt_model_path'):
            try:
                filename = Path(self.model_source.pt_model_path).name
                self.model_source.pt_model_path = get_model_path(filename)
            except FileNotFoundError:
                pass
        return self

    @model_validator(mode='after')
    def validate_input_sizes_match(self) -> 'YOLOConfig':
        """validates input sizes match between inference and engine settings"""
        return self  #both should use the same inpute size


class FaceMeshConfig(BaseModel):
    """Mediapipe face mesh configs"""
    max_num_faces: int = Field(default=4, ge=1, le=20, description="Max number of faces")
    refine_landmarks: bool = True
    min_detection_confidence: float =Field(default=0.5, ge=0.1, le=1.0, description="Minimum detection confidence")
    min_tracking_confidence: float = Field(default=0.5, ge=0.1, le=1.0, description="Minimum tracking confidence")
    static_image_mode: bool = False

class FaceDetectionConfig(BaseModel):
    """Mediapipe face detection configs"""
    model_selection: Literal[0, 1] = 0
    min_detection_confidence: float = Field(default=0.5, ge=0.1, le=1.0, description="Minimum face detection confidence")

class MediaPipeConfig(BaseModel):
    """Mediapipe configs"""
    face_mesh: FaceMeshConfig = Field(default_factory=FaceMeshConfig, description="FaceMesh config")
    face_detection: FaceDetectionConfig = Field(default_factory=FaceDetectionConfig, description="FaceDetection config")

class CanonConfig(BaseModel):
    """Canon configs"""
    host: str = "127.0.0.1"
    port: int = Field(default=8089, ge=1024, le=65535)
    timeout: float = Field(default=5.0, gt=0)
    frame_queue_size: int = Field(default=5, ge=1, le=20)
    reconnect_attempts: int = Field(default=3, ge=1, le=10)
    reconnect_delay: float = Field(default=2.0, gt=0)

class DepthRange(BaseModel):
    """Depth range configs"""
    min: float = Field(default=0.4, gt=0)
    max: float = Field(default=8.0, gt=0)

    @model_validator(mode='after')
    def validate_range(self) -> 'DepthRange':
        if self.max <= self.min:
            raise ValueError(f'max({self.min}) must be greater than min({self.max})')
        return self

class RealSenseConfig(BaseModel):
    """RealSense depth processing configs"""
    depth_topic: str = "/camera/aligned_depth_to_color/image_raw"
    camera_info_topic: str = "/camera/aligned_depth_to_color/camera_info"
    depth_range: DepthRange = Field(default_factory=DepthRange)
    roi_percentage: float = Field(default=0.4, gt=0, le=1.0)
    outlier_rejection_sigma: float = Field(default=2.0, gt=0)
    min_valid_pixels_ratio: float = Field(default=0.5, gt=0, le=1.0)

class VisionConfig(BaseModel):
    """Vision pipeline configs"""
    yolo: YOLOConfig = Field(default_factory=YOLOConfig)
    mediapipe: MediaPipeConfig = Field(default_factory=MediaPipeConfig)
    canon: CanonConfig = Field(default_factory=CanonConfig)
    realsense: RealSenseConfig = Field(default_factory=RealSenseConfig)

class PerformanceConfig(BaseModel):
    """Performance and resources configs"""
    target_fps: int = Field(default=30, ge=1, le=60)
    processing_timeout: float = Field(default=0.1, gt=0)
    max_cpu_usage: int = Field(default=80, ge=10, le=100)
    max_gpu_usage: int = Field(default=85, ge=10, le=100)
    max_memory_usage: int = Field(default=12, ge=1, le=64)

class LoggingConfig(BaseModel):
    """Logging configs"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["structured", "simple"] = "structured"
    file_path: str = "logs/manriix_photo_va.log"
    max_file_size_mb: int = Field(default=100, ge=1)
    backup_count: int = Field(default=5, ge=1)
    console_output: bool = True

class ThreadingConfig(BaseModel):
    """Threading configs"""
    max_workers: int = Field(default=4, ge=1, le=16)
    thread_timeout: float = Field(default=30.0, gt=0)
    use_async: bool = True

class PerformanceBenchmarks(BaseModel):
    """Performance benchmarks"""
    yolo_inference_ms: int = Field(default=15, ge=0)
    face_analysis_ms: int = Field(default=10, ge=0)
    depth_processing_ms: int = Field(default=5, ge=0)

class TestingConfig(BaseModel):
    """Testing configs"""
    test_image_path: str = "tests/test_data/test_image.jpg"
    test_depth_path: str = "tests/test_data/test_depth.png"
    mock_canon_server: bool = True
    performance_benchmarks: PerformanceBenchmarks = Field(default_factory=PerformanceBenchmarks)

class SystemConfig(BaseModel):
    """System configs"""
    name: str = "manriix_photo_capture_va"
    version: str = "1.0.0"
    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = True

class CameraMounting(BaseModel):
    """Camera mounting configs"""
    height_from_ground: float = Field(1.68, gt=0, description="Camera height in meters")
    tilt_angle_deg: float = Field(0.0, ge=-45, le=45, description="Camera tilt angle")

class PositioningFiltering(BaseModel):
    """Positioning filtering configs"""
    enabled: bool = True
    median_window_size: int = Field(default=5, ge=1, le=20, description="Window size for median filtering")
    max_position_change_per_frame: float = Field(default=0.2, gt=0, description="Max position change per frame in meters")
    outlier_rejection_sigma: float = Field(default=2.0, gt=0, description="Outlier rejection sigma")

class FrameConfig(BaseModel):
    """Coordinate frame names for TF lookups"""
    camera_optical: str = Field(
        default="camera_color_optical_frame",
        description="Camera optical frame name"
    )
    camera_link: str = Field(
        default="camera_link",
        description="Camera link frame name"
    )
    base_link: str = Field(
        default="base_link",
        description="Robot base link frame name"
    )
    base_footprint: str = Field(
        default="base_footprint",
        description="Robot base footprint frame name"
    )

class GroundPlane(BaseModel):
    """Ground plane detection configs"""
    detection_method: str = "fixed_height"  # or "ransac"

class PositioningConfig(BaseModel):
    """3D Positioning configs"""
    coordinate_system: str = "camera_frame"
    camera_mounting: CameraMounting = Field(default_factory=CameraMounting)
    positioning_filtering: PositioningFiltering = Field(default_factory=PositioningFiltering)
    ground_plane: GroundPlane = Field(default_factory=GroundPlane)
    frames: FrameConfig = Field(
        default_factory=FrameConfig,
        description="Coordinate frame names"
    )
    use_dynamic_transforms: bool = Field(
        default=True,
        description="Use ROS2 TF for dynamic transforms"
    )
    transform_cache_duration: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Transform cache duration in seconds"
    )

class GimbalConfig(BaseModel):
    """Gimbal configs"""
    control_method: Literal["action", "direct"] = "direct"
    controller_name: str = "gimbal_controller"
    joint_names: List[str] = ["joint_5", "joint_6", "joint_7"]

    class Limits(BaseModel):
        yaw_range: Tuple[float, float] = (-1.57, 1.57)
        pitch_range: Tuple[float, float] = (-0.785, 0.785)
        roll_range: Tuple[float, float] = (0.0, 0.0)

    class Movement(BaseModel):
        max_velocity: float = Field(0.5, gt=0)
        default_move_time: float = Field(1.0, gt=0)

    limits: Limits = Limits()
    movement: Movement = Movement()

class FocusConfig(BaseModel):
    """Focus configs"""
    control_topic: str = "/set_focus_position"
    motor_range: Tuple[int, int] = (0, 4095)
    default_focal_length: int = Field(50, gt=0)
    calculation_method: Literal["hyperfocal", "calibration_table"] = "hyperfocal"
    hyperfocal_distance: float = Field(15.0, gt=0)

class CameraControlConfig(BaseModel):
    """Camera control configs"""
    gimbal: GimbalConfig = Field(default_factory=GimbalConfig)
    focus: FocusConfig = Field(default_factory=FocusConfig)

class WorkflowConfig(BaseModel):
    """Workflow configs"""
    max_session_time_s: float = Field(60.0, gt=0)
    max_positioning_time_s: float = Field(30.0, gt=0)
    capture_countdown_s: float = Field(3.0, gt=0)
    photos_per_session: int = Field(1, ge=1, le=10)

class ActivationConfig(BaseModel):
    """Activation configs"""
    trigger_topic: str = "/va_photo_capture"
    trigger_message: str = "y"
    auto_start: bool = False


class SceneClassificationConfig(BaseModel):
    """Scene classification configuration"""
    enabled: bool = True

    class DistanceRange(BaseModel):
        """Distance range for a scene type"""
        min_distance: float
        max_distance: float
        optimal_distance: float
        max_horizontal_spread: Optional[float] = None

    class CompositionPadding(BaseModel):
        """Composition padding for framing"""
        horizontal: float
        vertical: float

    class CompositionPaddingConfig(BaseModel):
        """Composition padding for all scene types"""
        portrait: 'SceneClassificationConfig.CompositionPadding'
        couple: 'SceneClassificationConfig.CompositionPadding'
        small_group: 'SceneClassificationConfig.CompositionPadding'
        medium_group: 'SceneClassificationConfig.CompositionPadding'
        large_group: 'SceneClassificationConfig.CompositionPadding'

    class DistanceRanges(BaseModel):
        """Distance ranges for all scene types"""
        portrait: 'SceneClassificationConfig.DistanceRange'
        couple: 'SceneClassificationConfig.DistanceRange'
        small_group: 'SceneClassificationConfig.DistanceRange'
        medium_group: 'SceneClassificationConfig.DistanceRange'
        large_group: 'SceneClassificationConfig.DistanceRange'

    distance_ranges: DistanceRanges
    composition_padding: Dict[str, CompositionPaddingConfig] = Field(
        default_factory=dict,
        description="Composition padding configuration"
    )

class PhotoCaptureConfig(BaseModel):
    """Photo capture configs"""
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)
    scene_classification: SceneClassificationConfig = Field(default_factory=SceneClassificationConfig)


class ROS2TopicsConfig(BaseModel):
    """ROS2 topics configs"""
    # Input topics
    realsense_depth: str = "/camera/aligned_depth_to_color/image_raw"
    realsense_camera_info: str = "/camera/aligned_depth_to_color/camera_info"
    joint_states: str = "/joint_states"
    activation_trigger: str = "/va_photo_capture"

    # Output topics
    gimbal_commands: str = "/gimbal_controller/commands"
    focus_commands: str = "/set_focus_position"
    capture_trigger: str = "/capture_image_topic"
    tts_output: str = "/tts_text_output"

    # Status topics
    system_status: str = "/photo_capture/status"
    detection_debug: str = "/photo_capture/detections"

class ControlTimingConfig(BaseModel):
    """Control timing configs"""
    main_loop_hz: float = Field(5.0, gt=0, le=30)
    vision_processing_hz: float = Field(10.0, gt=0, le=60)
    depth_processing_hz: float = Field(5.0, gt=0, le=30)
    position_update_hz: float = Field(5.0, gt=0, le=30)

class Settings(BaseModel):
    """Main settings class"""

    model_config = ConfigDict(extra="ignore")

    system: SystemConfig = Field(default_factory=SystemConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    threading: ThreadingConfig = Field(default_factory=ThreadingConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)
    positioning: PositioningConfig = Field(default_factory=PositioningConfig)
    camera_control: CameraControlConfig = Field(default_factory=CameraControlConfig)
    photo_capture: PhotoCaptureConfig = Field(default_factory=PhotoCaptureConfig)
    ros2_topics: ROS2TopicsConfig = Field(default_factory=ROS2TopicsConfig)
    control_timing: ControlTimingConfig = Field(default_factory=ControlTimingConfig)


    @classmethod
    def from_yaml(cls, config_path: str) -> 'Settings':
        """Load settings from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML config: {exc}")

        if not isinstance(config_data, dict):
            raise ValueError(f"Config must be a dict: {type(config_data)}")
        return cls(**config_data)

    def to_yaml(self, output_path: str) -> None:
        """Save current settings to YAML file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f,  default_flow_style=False, indent=2)

    @model_validator(mode='after')
    def validate_vision_config(self) -> 'Settings':
        """Validate vision config"""
        model_path = Path(self.vision.yolo.model_path)
        if not model_path.exists() and self.system.environment == "production":
            raise ValueError(f"YOLO model not found: {model_path}")
        return self
#========================================================================
#                   photo_capture.yaml validations
# =======================================================================
class StateConfig(BaseModel):
    """Individual state configs"""
    description: str
    timeout_s: Optional[float] = None
    actions: Optional[List[str]] = None
    min_detection_time_s: Optional[float] = None
    min_person_confidence: Optional[float] = None
    max_guidance_attempts: Optional[int] = None
    position_tolerance_m: Optional[float] = None
    required_checks: Optional[List[str]] = None
    duration_s: Optional[float] = None
    display_time_s: Optional[float] = None

class WorkflowStatesConfig(BaseModel):
    """Workflow states configs"""
    idle: StateConfig
    initializing: StateConfig
    detecting: StateConfig
    positioning: StateConfig
    adjusting_camera: StateConfig
    verifying: StateConfig
    countdown: StateConfig
    capturing: StateConfig
    complete: StateConfig

class WorkflowStatesWrapperConfig(BaseModel):
    """Workflow configs"""
    states: WorkflowStatesConfig = Field(default_factory=WorkflowStatesConfig)

class VoiceGuidanceMessages(BaseModel):
    """Voice guidance messages configs"""
    welcome: Union[str, List[str]]
    single_person_detected: Union[str, List[str]]
    couple_detected: Union[str, List[str]]
    group_detected: Union[str, List[str]]
    move_closer: Union[str, List[str]]
    move_further: Union[str, List[str]]
    move_left: Union[str, List[str]]
    move_right: Union[str, List[str]]
    move_together: Union[str, List[str]]
    spread_out: Union[str, List[str]]
    perfect_position: Union[str, List[str]]
    countdown_start: Union[str, List[str]]
    countdown_numbers: List[str]
    capture_complete: Union[str, List[str]]
    timeout_warning: Union[str, List[str]]

class VoiceGuidanceTiming(BaseModel):
    """voice guidance timings"""
    message_interval_s: float = 2.0
    position_update_interval_s: float = 3.0

class SubjectPositioning(BaseModel):
    """subject positioning preferences"""
    vertical_center_offset: float = 0.1
    horizontal_tolerance: float = 0.05

class QualityChecks(BaseModel):
    """Quality checks requirements"""
    min_face_size_pixels: int = 80
    max_motion_blur_threshold: float = 5.0
    min_contrast_level: float = 0.3

class VoiceGuidanceConfig(BaseModel):
    """VoiceGuidance configs"""
    language: str = "en"
    messages: VoiceGuidanceMessages = Field(default_factory=VoiceGuidanceMessages)
    timing: VoiceGuidanceTiming = Field(default_factory=VoiceGuidanceTiming)
    subject_positioning: SubjectPositioning = Field(default_factory=SubjectPositioning)
    quality_checks: QualityChecks = Field(default_factory=QualityChecks)

class DepthFailures(BaseModel):
    """Depth failures handling"""
    max_consecutive_failures: int = 5
    fallback_to_estimation: bool = True
    estimation_method: str = "bbox_scaling"

class DetectionFailures(BaseModel):
    """Detection failures handling"""
    max_no_detection_time_s: float = 5.0
    recovery_actions: List[str]

class ControlFailures(BaseModel):
    """Control failures handling"""
    gimbal_timeout_recovery: str = "return_to_home"
    focus_timeout_recovery: str = "use_hyperfocal"
    max_recovery_attempts: int = 3

class SystemFailures(BaseModel):
    """System failure handling"""
    critical_component_timeout_s: float = 10.0
    emergency_shutdown_enabled: bool = True
    auto_restart_attempts: int = 2

class ErrorHandlingConfig(BaseModel):
    """Error handling configs"""
    depth_failures: DepthFailures = Field(default_factory=DepthFailures)
    detection_failures: DetectionFailures = Field(default_factory=DetectionFailures)
    control_failures: ControlFailures = Field(default_factory=ControlFailures)
    system_failures: SystemFailures = Field(default_factory=SystemFailures)

class PhotoWorkflowConfig(BaseModel):
    """Photo workflow configs """

    model_config = ConfigDict(extra="ignore")

    workflow: WorkflowStatesWrapperConfig
    voice_guidance: VoiceGuidanceConfig
    error_handling: ErrorHandlingConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "PhotoWorkflowConfig":
        """Load workflow config from YAML file"""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Workflow config file does not exist: {config_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML config: {exc}")

        if not isinstance(config_data, dict):
            raise ValueError(f"Config must be a dict: {type(config_data)}")

        return cls(**config_data)

# =======================================================================
#                       exposure_config.yaml validations
# =======================================================================

class SensorSpecs(BaseModel):
    """Camera sensor specifications"""
    type: Literal["full_frame", "aps_c", "micro_four_thirds"] = "full_frame"
    width_mm: float = Field(gt=0, description="Sensor width in mm")
    height_mm: float = Field(gt=0, description="Sensor height in mm")
    crop_factor: float = Field(gt=0, le=2, description="Crop factor (1.0 for full-frame)")
    circle_of_confusion_mm: float = Field(gt=0, le=0.1, description="Circle of confusion in mm")
    megapixels: float = Field(gt=0, description="Sensor megapixels")
    resolution_width: int = Field(gt=0, description="Horizontal resolution in pixels")
    resolution_height: int = Field(gt=0, description="Vertical resolution in pixels")

    @field_validator('crop_factor')
    @classmethod
    def validate_crop_factor(cls, v: float, info) -> float:
        """Ensure crop factor matches sensor type"""
        sensor_type = info.data.get('type')
        if sensor_type == "full_frame" and v != 1.0:
            raise ValueError("Full-frame sensors must have crop_factor = 1.0")
        return v


class LensSpecs(BaseModel):
    """Camera lens specifications"""
    model: str = Field(description="Lens model name")
    focal_length_min: int = Field(gt=0, description="Minimum focal length in mm")
    focal_length_max: int = Field(gt=0, description="Maximum focal length in mm")

    # Aperture range (f-numbers) | f-number = wider opening, larger f-number = smaller opening
    f_number_min: float = Field(gt=0, description="Minimum f-number (widest aperture)")
    f_number_max: float = Field(gt=0, description="Maximum f-number (smallest aperture)")

    available_apertures: List[float] = Field(description="Available f-stop values")
    optical_is_stops: Optional[float] = Field(default=None, description="Optical IS in stops")
    ibis_coordinated_stops: Optional[float] = Field(default=None, description="IBIS coordinated stops")

    @field_validator('focal_length_max')
    @classmethod
    def validate_focal_range(cls, v: int, info) -> int:
        """ensure max focal length > min"""
        min_fl = info.data.get('focal_length_min')
        if min_fl and v <= min_fl:
            raise ValueError(f"focal_length_max ({v}) must be greater than focal_length_min ({min_fl})")
        return v

    @field_validator('f_number_max')
    @classmethod
    def validate_aperture_range(cls, v: float, info) -> float:
        """ensure max f-number > min f-number (smaller opening > larger opening)"""
        min_fn = info.data.get('f_number_min')
        if min_fn and v <= min_fn:
            raise ValueError(
                f"f_number_max ({v}) must be greater than f_number_min ({min_fn}). "
                f"Remember: larger f-number = smaller aperture opening"
            )
        return v

    @field_validator('available_apertures')
    @classmethod
    def validate_available_apertures(cls, v: List[float], info) -> List[float]:
        """ensure available apertures are within lens range and sorted"""
        if not v:
            raise ValueError("available_apertures cannot be empty")

        min_fn = info.data.get('f_number_min')
        max_fn = info.data.get('f_number_max')

        if min_fn and max_fn:
            for ap in v:
                if ap < min_fn or ap > max_fn:
                    raise ValueError(
                        f"Aperture f/{ap} outside lens range "
                        f"[f/{min_fn} - f/{max_fn}]"
                    )

        # Ensure sorted
        if v != sorted(v):
            raise ValueError("available_apertures must be sorted in ascending order")

        return v


class ISOSpecs(BaseModel):
    """ISO specs"""
    min: int = Field(gt=0, description="Minimum ISO")
    max: int = Field(gt=0, description="Maximum ISO")
    expanded_min: Optional[int] = Field(default=None, description="Expanded ISO minimum")
    expanded_max: Optional[int] = Field(default=None, description="Expanded ISO maximum")
    standard_values: List[int] = Field(description="Available ISO values")

    @field_validator('max')
    @classmethod
    def validate_iso_range(cls, v: int, info) -> int:
        """ensure max ISO > min ISO"""
        min_iso = info.data.get('min')
        if min_iso and v <= min_iso:
            raise ValueError(f"max ISO ({v}) must be greater than min ISO ({min_iso})")
        return v

    @field_validator('standard_values')
    @classmethod
    def validate_standard_values(cls, v: List[int], info) -> List[int]:
        """ensure standard ISO values are within range and sorted"""
        if not v:
            raise ValueError("standard_values cannot be empty")

        min_iso = info.data.get('min')
        max_iso = info.data.get('max')

        if min_iso and max_iso:
            for iso in v:
                if iso < min_iso or iso > max_iso:
                    raise ValueError(f"ISO {iso} outside camera range [{min_iso}-{max_iso}]")

        # Ensure sorted
        if v != sorted(v):
            raise ValueError("standard_values must be sorted in ascending order")

        return v

class CameraSpecs(BaseModel):
    """Camera specs"""
    sensor: SensorSpecs = Field(default_factory=SensorSpecs, description="Sensor specs")
    lens: LensSpecs = Field(default_factory=LensSpecs, description="Lens specs")
    iso: ISOSpecs = Field(default_factory=ISOSpecs, description="ISO specs")

class ExposureControlSettings(BaseModel):
    """Exposure control settings"""
    enabled: bool = Field(default=False, description="Enable manual exposure control")
    strategy: Literal["balanced", "portrait", "sports", "landscape"] = "balanced"
    use_frame_analysis: bool = Field(default=True, description="Use vision-based lighting analysis")
    log_recommendations: bool = Field(default=True, description="Log exposure recommendations")


class ManualSettings(BaseModel):
    """Manual default settings when exposure control disabled"""
    default_aperture: float = Field(gt=0, description="Default aperture value")
    default_shutter: str = Field(description="Default shutter speed")
    default_iso: int = Field(gt=0, description="Default ISO value")
    recommended_camera_mode: str = Field(default="Auto", description="Recommended camera mode")

    @field_validator('default_shutter')
    @classmethod
    def validate_shutter_format(cls, v: str) -> str:
        """Validate shutter speed format"""
        if not v:
            raise ValueError("default_shutter cannot be empty")

        # Accept formats like "1/125", "1/250", "1", "2"
        if '/' in v:
            parts = v.split('/')
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"Invalid shutter speed format: {v}. Use format like '1/125'")
        elif not v.replace('.', '').isdigit():
            raise ValueError(f"Invalid shutter speed format: {v}")

        return v


class ApertureRangeSettings(BaseModel):
    """Aperture range for a scene type"""
    min: float = Field(gt=0, description="Minimum aperture")
    max: float = Field(gt=0, description="Maximum aperture")
    preferred: float = Field(gt=0, description="Preferred aperture")
    reason: str = Field(description="Reasoning for this range")

    @model_validator(mode='after')
    def validate_range(self) -> 'ApertureRangeSettings':
        """Ensure min <= preferred <= max"""
        if not (self.min <= self.preferred <= self.max):
            raise ValueError(f"Aperture range invalid: min={self.min}, preferred={self.preferred}, max={self.max}")
        return self


class ApertureRanges(BaseModel):
    """Aperture ranges for all scene types"""
    portrait_single: ApertureRangeSettings
    portrait_couple: ApertureRangeSettings
    group_small: ApertureRangeSettings
    group_large: ApertureRangeSettings
    landscape: ApertureRangeSettings
    action: ApertureRangeSettings


class ShutterSpeedPreferences(BaseModel):
    """Shutter speed preferences"""
    min_handheld: str = Field(description="Minimum handheld shutter speed")
    min_stabilized: str = Field(description="Minimum with stabilization")
    portrait_min: str = Field(description="Minimum for portraits")
    portrait_close_min: str = Field(description="Minimum for close portraits")
    action_min: str = Field(description="Minimum for action")
    sports_min: str = Field(description="Minimum for sports")
    focal_length_rule_multiplier: int = Field(gt=0, description="Multiplier for 1/focal_length rule")


class ISOPreferences(BaseModel):
    """ISO preferences"""
    preferred_base: int = Field(gt=0, description="Preferred base ISO")
    acceptable_max: int = Field(gt=0, description="Maximum acceptable ISO")
    emergency_max: int = Field(gt=0, description="Emergency maximum ISO")
    portrait_max: int = Field(gt=0, description="Maximum for portraits")
    landscape_max: int = Field(gt=0, description="Maximum for landscapes")
    action_max: int = Field(gt=0, description="Maximum for action")

    @model_validator(mode='after')
    def validate_iso_hierarchy(self) -> 'ISOPreferences':
        """Ensure ISO limits are sensible"""
        if not (self.preferred_base < self.acceptable_max < self.emergency_max):
            raise ValueError("ISO hierarchy invalid: base < acceptable < emergency")
        return self


class DistanceThresholds(BaseModel):
    """Subject distance thresholds in meters"""
    close_portrait: float = Field(gt=0, description="Close portrait distance")
    standard_portrait: float = Field(gt=0, description="Standard portrait distance")
    group_distance: float = Field(gt=0, description="Group distance")
    far_distance: float = Field(gt=0, description="Far distance")


class PhotoPreferences(BaseModel):
    """Photography preferences"""
    aperture_ranges: ApertureRanges
    shutter_speed: ShutterSpeedPreferences
    iso: ISOPreferences
    distance_thresholds: DistanceThresholds


class LightingThreshold(BaseModel):
    """Lighting condition threshold"""
    min: Optional[int] = Field(default=None, ge=0, le=255)
    max: Optional[int] = Field(default=None, ge=0, le=255)
    recommended_iso: int = Field(gt=0)
    recommended_aperture: float = Field(gt=0)

    @model_validator(mode='after')
    def validate_min_max(self) -> 'LightingThreshold':
        """Ensure min < max if both present"""
        if self.min is not None and self.max is not None:
            if self.min >= self.max:
                raise ValueError(f"min ({self.min}) must be less than max ({self.max})")
        return self


class LightingThresholds(BaseModel):
    """All lighting condition thresholds"""
    very_dark: LightingThreshold
    dark: LightingThreshold
    moderate: LightingThreshold
    bright: LightingThreshold
    very_bright: LightingThreshold


class HistogramSettings(BaseModel):
    """Histogram analysis settings"""
    use_histogram: bool = True
    contrast_threshold: int = Field(ge=0, le=255, description="Contrast threshold")


class LightingAnalysisSettings(BaseModel):
    """Lighting analysis configuration"""
    thresholds: LightingThresholds
    histogram: HistogramSettings


class CompositionWeights(BaseModel):
    """Composition scoring weights"""
    rule_of_thirds: float = Field(ge=0, le=1)
    balance: float = Field(ge=0, le=1)
    subject_placement: float = Field(ge=0, le=1)
    negative_space: float = Field(ge=0, le=1)

    @model_validator(mode='after')
    def validate_sum(self) -> 'CompositionWeights':
        """Ensure weights sum to 1.0"""
        total = (self.rule_of_thirds + self.balance +
                 self.subject_placement + self.negative_space)
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Composition weights must sum to 1.0, got {total}")
        return self


class CompositionThresholds(BaseModel):
    """Composition score thresholds"""
    minimum_score: float = Field(ge=0, le=1, description="Minimum acceptable score")
    excellent_score: float = Field(ge=0, le=1, description="Excellent score threshold")

    @model_validator(mode='after')
    def validate_thresholds(self) -> 'CompositionThresholds':
        """Ensure minimum < excellent"""
        if self.minimum_score >= self.excellent_score:
            raise ValueError(f"minimum_score must be less than excellent_score")
        return self


class RuleOfThirdsSettings(BaseModel):
    """Rule of thirds settings"""
    tolerance_pixels: int = Field(ge=0, description="Tolerance in pixels")


class BalanceSettings(BaseModel):
    """Balance settings"""
    optimal_deviation: float = Field(ge=0, le=0.5, description="Optimal off-center ratio")


class EdgeSafetySettings(BaseModel):
    """Edge safety settings"""
    threshold: float = Field(ge=0, le=0.5, description="Edge safety threshold")


class CompositionSettings(BaseModel):
    """Composition analysis configuration"""
    enabled: bool = True
    weights: CompositionWeights
    thresholds: CompositionThresholds
    rule_of_thirds: RuleOfThirdsSettings
    balance: BalanceSettings
    edge_safety: EdgeSafetySettings

class DepthOfFieldPreferences(BaseModel):
    """DOF preferences"""
    portrait_dof_min: float = Field(gt=0, description="Minimum portrait DOF")
    portrait_dof_max: float = Field(gt=0, description="Maximum portrait DOF")
    group_dof_min: float = Field(gt=0, description="Minimum group DOF")
    landscape_dof_target: str = Field(description="Landscape DOF target")

class DepthOfFieldConfig(BaseModel):
    """Depth of field configuration"""
    preferences: DepthOfFieldPreferences

class ValidationSettings(BaseModel):
    """Validation configuration"""
    strict_mode: bool = Field(default=True, description="Strict validation mode")
    warn_on: Dict[str, bool] = Field(description="Warning triggers")
    auto_correct: bool = Field(default=True, description="Auto-correct invalid settings")
    auto_correct_to_nearest: bool = Field(default=True, description="Round to nearest valid")


class FeatureFlags(BaseModel):
    """Feature flags for development"""
    auto_framing: bool = True
    composition_analysis: bool = True
    person_tracking: bool = True
    multi_shot_capture: bool = False
    hdr_bracketing: bool = False
    focus_stacking: bool = False
    ai_composition_suggestions: bool = False
    scene_recognition: bool = False


class LoggingSettings(BaseModel):
    """Logging configuration"""
    log_exposure_calculations: bool = True
    log_composition_scores: bool = True
    log_lighting_analysis: bool = True
    save_debug_frames: bool = False
    debug_output_path: str = "/tmp/manriix_debug"
    log_calculation_time: bool = True

    @field_validator('debug_output_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path is valid"""
        if not v:
            raise ValueError("debug_output_path cannot be empty")
        return v


class ExposureConfig(BaseModel):
    """Complete exposure configuration with validation"""
    exposure_control: ExposureControlSettings
    camera_specs: CameraSpecs
    manual_settings: ManualSettings
    preferences: PhotoPreferences
    lighting_analysis: LightingAnalysisSettings
    composition: CompositionSettings
    validation: ValidationSettings
    features: FeatureFlags
    depth_of_field: DepthOfFieldConfig
    logging: LoggingSettings

    @classmethod
    def from_yaml(cls, config_path: str) -> "ExposureConfig":
        """Load exposure config from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Exposure config file does not exist: {config_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML config: {exc}")

        if not isinstance(config_data, dict):
            raise ValueError(f"Config must be a dict: {type(config_data)}")

        return cls(**config_data)

    @model_validator(mode='after')
    def validate_exposure_config(self) -> 'ExposureConfig':
        """Cross-validate settings"""
        # Validate manual defaults against camera capabilities
        camera_specs = self.camera_specs
        manual_settings = self.manual_settings

        # Check aperture (using new naming)
        f_min = camera_specs.lens.f_number_min
        f_max = camera_specs.lens.f_number_max
        default_ap = manual_settings.default_aperture

        if not (f_min <= default_ap <= f_max):
            raise ValueError(
                f"default_aperture f/{default_ap} outside lens range "
                f"[f/{f_min} - f/{f_max}]"
            )

        # Check ISO
        if not (camera_specs.iso.min <= manual_settings.default_iso <= camera_specs.iso.max):
            raise ValueError(
                f"default_iso {manual_settings.default_iso} outside camera range "
                f"[{camera_specs.iso.min}-{camera_specs.iso.max}]"
            )

        # Validate preference ISOs against camera capabilities
        iso_prefs = self.preferences.iso
        if iso_prefs.emergency_max > camera_specs.iso.max:
            raise ValueError(
                f"emergency_max ISO {iso_prefs.emergency_max} exceeds camera max {camera_specs.iso.max}"
            )

        return self


#====================================================================================

#global settings  --> config.yaml
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get global settings
    :rtype: Settings
    """
    global _settings
    if _settings is None:
        config_path = get_config_path('config.yaml', 'CONFIG_PATH')
        _settings = Settings.from_yaml(config_path)
    return _settings

def reload_settings(config_path: Optional[str] = None) -> Settings:
    """Reload global settings"""
    global _settings
    if config_path is None:
        config_path =  get_config_path('config.yaml', 'CONFIG_PATH')
    _settings = Settings.from_yaml(config_path)
    return _settings

def get_development_settings() -> Settings:
    """Get development settings"""
    settings = get_settings()
    if settings.system.environment == "development":
        settings = settings.model_copy(update={
            "logging": settings.logging.model_copy(update={"level": "DEBUG"}),
            "system": settings.system.model_copy(update={"debug": True}),
            "testing": settings.testing.model_copy(update={"mock_canon_server": True})
        })
    return settings

def get_production_settings() -> Settings:
    """Get production settings"""
    settings = get_settings()
    if settings.system.environment == "production":
        settings = settings.model_copy(update={
            "logging": settings.logging.model_copy(update={"level": "INFO", "console_output": False}),
            "system": settings.system.model_copy(update={"debug": False}),
            "testing": settings.testing.model_copy(update={"mock_canon_server": False})
        })
    return settings


# #global settings  --> photo_capture.yaml

_workflow_config: Optional[PhotoWorkflowConfig] = None


def get_workflow_config() -> PhotoWorkflowConfig:
    """Get global photo workflow config"""
    global _workflow_config
    if _workflow_config is None:
        config_path = get_config_path('photo_capture.yaml', 'WORKFLOW_CONFIG_PATH')
        _workflow_config = PhotoWorkflowConfig.from_yaml(config_path)
    return _workflow_config


def reload_workflow_config(config_path: Optional[str] = None) -> PhotoWorkflowConfig:
    """Reload photo workflow config"""
    global _workflow_config
    if config_path is None:
        config_path = get_config_path('photo_capture.yaml', 'WORKFLOW_CONFIG_PATH')
    _workflow_config = PhotoWorkflowConfig.from_yaml(config_path)
    return _workflow_config


# #global settings  --> exposure_config.yaml

_exposure_config: Optional[ExposureConfig] = None

def get_exposure_config() -> ExposureConfig:
    """Get global exposure config"""
    global _exposure_config
    if _exposure_config is None:
        exposure_config_path = get_config_path('exposure_config.yaml', 'EXPOSURE_CONFIG_PATH')
        _exposure_config = ExposureConfig.from_yaml(exposure_config_path)
    return _exposure_config

def reload_exposure_config(config_path: Optional[str] = None) -> ExposureConfig:
    """Reload exposure config"""
    global _exposure_config
    if config_path is None:
        config_path = get_config_path('exposure_config.yaml', 'EXPOSURE_CONFIG_PATH')
    _exposure_config = ExposureConfig.from_yaml(config_path)
    return _exposure_config



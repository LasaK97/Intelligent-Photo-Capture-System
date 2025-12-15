import os
from pathlib import Path
from typing import List, Tuple, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings
import yaml

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
        return v

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

class PositioningConfig(BaseModel):
    """3D Positioning configs"""
    coordinate_system: str = "camera_frame"
    camera_mounting: CameraMounting = Field(default_factory=CameraMounting)
    positioning_filtering: PositioningFiltering = Field(default_factory=PositioningFiltering)

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

    ## Photo capture.yaml validations

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


#global settings  --> config.yaml
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get global settings
    :rtype: Settings
    """
    global _settings
    if _settings is None:
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
        _settings = Settings.from_yaml(config_path)
    return _settings

def reload_settings(config_path: Optional[str] = None) -> Settings:
    """Reload global settings"""
    global _settings
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
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
        config_path = os.getenv("WORKFLOW_CONFIG_PATH", "config/photo_capture.yaml")
        _workflow_config = PhotoWorkflowConfig.from_yaml(config_path)
    return _workflow_config


def reload_workflow_config(config_path: Optional[str] = None) -> PhotoWorkflowConfig:
    """Reload photo workflow config"""
    global _workflow_config
    if config_path is None:
        config_path = os.getenv("WORKFLOW_CONFIG_PATH", "config/photo_capture.yaml")
    _workflow_config = PhotoWorkflowConfig.from_yaml(config_path)
    return _workflow_config

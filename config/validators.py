from typing import Dict, Any, List, Tuple, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path


# =============================================================================
# HARDWARE DOMAIN VALIDATORS
# =============================================================================

# -----------------------------------------------------------------------------
# Camera Configuration Validators
# -----------------------------------------------------------------------------

class FrameClientConfig(BaseModel):
    """Canon frame client configuration"""
    host: str = Field(pattern=r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$|^localhost$|^127\.0\.0\.1$')
    port: int = Field(ge=1024, le=65535)
    timeout: float = Field(gt=0, le=30)
    frame_queue_size: int = Field(ge=1, le=20)
    reconnect_attempts: int = Field(ge=1, le=10)
    reconnect_delay: float = Field(gt=0, le=10)


class ExposureControlConfig(BaseModel):
    """Camera exposure control settings"""
    enabled: bool = False
    strategy: Literal["balanced", "portrait", "sports", "landscape"] = "balanced"
    use_frame_analysis: bool = True
    log_recommendations: bool = True


class ManualSettingsConfig(BaseModel):
    """Manual exposure settings"""
    default_aperture: float = Field(ge=1.4, le=22.0)
    default_shutter: str = Field(pattern=r'^1/\d+$|^\d+$')  # e.g., "1/125" or "1"
    default_iso: int = Field(ge=100, le=102400)
    recommended_camera_mode: str = "Auto"

    @field_validator('default_iso')
    @classmethod
    def validate_iso_value(cls, v: int) -> int:
        """Validate ISO is in standard values"""
        valid_isos = [100, 125, 160, 200, 250, 320, 400, 500, 640, 800,
                    1000, 1250, 1600, 2000, 2500, 3200, 4000, 5000, 6400,
                    8000, 10000, 12800, 16000, 20000, 25600, 32000, 40000,
                    51200, 64000, 80000, 102400]
        if v not in valid_isos:
            raise ValueError(f'ISO {v} not in standard values. Valid: {valid_isos[:5]}...{valid_isos[-3:]}')
        return v


class ShutterPreferencesConfig(BaseModel):
    """Shutter speed preferences"""
    min_handheld: str = Field(pattern=r'^1/\d+$')
    min_stabilized: str = Field(pattern=r'^1/\d+$')
    focal_length_rule_multiplier: float = Field(ge=1, le=3, default=2)
    portrait_min: str = Field(pattern=r'^1/\d+$')
    portrait_close_min: str = Field(pattern=r'^1/\d+$')
    action_min: str = Field(pattern=r'^1/\d+$')
    sports_min: str = Field(pattern=r'^1/\d+$')


class ISOPreferencesConfig(BaseModel):
    """ISO preferences"""
    preferred_base: int = Field(ge=50, le=400, default=100)
    acceptable_max: int = Field(ge=1600, le=25600, default=6400)
    emergency_max: int = Field(ge=3200, le=102400, default=12800)
    portrait_max: int = Field(ge=800, le=6400, default=3200)
    landscape_max: int = Field(ge=400, le=3200, default=1600)
    action_max: int = Field(ge=1600, le=12800, default=6400)

    @model_validator(mode='after')
    def validate_iso_hierarchy(self) -> 'ISOPreferencesConfig':
        """Ensure ISO values are in logical order"""
        if self.preferred_base > self.acceptable_max:
            raise ValueError('preferred_base must be <= acceptable_max')
        if self.acceptable_max > self.emergency_max:
            raise ValueError('acceptable_max must be <= emergency_max')
        return self


class ValidationConfig(BaseModel):
    """Camera validation settings"""
    strict_mode: bool = True
    warn_on: Dict[str, bool] = Field(
        default={
            'high_iso': True,
            'slow_shutter': True,
            'extreme_aperture': True
        }
    )
    auto_correct: bool = True
    auto_correct_to_nearest: bool = True


class CameraConfig(BaseModel):
    """Complete camera configuration"""
    frame_client: FrameClientConfig
    exposure_control: ExposureControlConfig
    manual_settings: ManualSettingsConfig
    shutter_preferences: ShutterPreferencesConfig
    iso_preferences: ISOPreferencesConfig
    validation: ValidationConfig

    model_config = {"extra": "forbid"}  # Don't allow extra fields


# -----------------------------------------------------------------------------
# Gimbal Configuration Validators
# -----------------------------------------------------------------------------

class GimbalControlConfig(BaseModel):
    """Gimbal control settings"""
    control_method: Literal["direct", "action"] = "direct"
    controller_name: str = Field(min_length=1)
    movement: Dict[str, Any]  # Will validate nested in next version

    @field_validator('movement')
    @classmethod
    def validate_movement(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate movement parameters"""
        required = {'default_move_time', 'safety_margin'}
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f'Missing required movement keys: {missing}')

        # Validate values
        if not 0.1 <= v['default_move_time'] <= 10.0:
            raise ValueError('default_move_time must be between 0.1 and 10.0 seconds')
        if not 0.0 <= v['safety_margin'] <= 0.5:
            raise ValueError('safety_margin must be between 0.0 and 0.5')

        return v


class MotionConstraintsConfig(BaseModel):
    """Gimbal motion constraints"""
    max_pan_velocity: float = Field(gt=0, le=180, default=30.0)  # degrees/second
    max_tilt_velocity: float = Field(gt=0, le=180, default=30.0)
    max_pan_acceleration: float = Field(gt=0, le=360, default=60.0)  # degrees/second²
    max_tilt_acceleration: float = Field(gt=0, le=360, default=60.0)
    max_jerk: float = Field(gt=0, le=1000, default=120.0)
    smoothness_factor: float = Field(ge=0.0, le=1.0, default=0.8)


class TrajectoryConfig(BaseModel):
    """Gimbal trajectory settings"""
    default_sample_rate: float = Field(ge=10.0, le=100.0, default=50.0)  # Hz
    min_trajectory_points: int = Field(ge=2, le=100, default=10)
    max_trajectory_points: int = Field(ge=10, le=10000, default=1000)
    profiles: Dict[str, bool] = Field(
        default={
            'linear': True,
            'ease_in_out': True,
            'ease_in': True,
            'ease_out': True,
            's_curve': True
        }
    )

    @model_validator(mode='after')
    def validate_trajectory_points(self) -> 'TrajectoryConfig':
        """Ensure min < max trajectory points"""
        if self.min_trajectory_points >= self.max_trajectory_points:
            raise ValueError('min_trajectory_points must be < max_trajectory_points')
        return self


class GimbalConfig(BaseModel):
    """Complete gimbal configuration"""
    gimbal_control: GimbalControlConfig
    motion_constraints: MotionConstraintsConfig
    trajectory: TrajectoryConfig

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Sensors Configuration Validators
# -----------------------------------------------------------------------------

class RealSenseProcessingConfig(BaseModel):
    """RealSense depth processing parameters"""
    max_processing_time_ms: int = Field(ge=1, le=100, default=15)
    roi_extraction_method: Literal["center_40_percent", "full_bbox", "adaptive"] = "center_40_percent"
    outlier_rejection_enabled: bool = True


class RealSenseConfig(BaseModel):
    """RealSense depth camera configuration"""
    processing: RealSenseProcessingConfig

    model_config = {"extra": "allow"}  # Allow additional fields from shared config


class FocusConfig(BaseModel):
    """Focus motor configuration"""
    control_topic: str = Field(pattern=r'^/[\w/]+$')  # Must start with /
    calculation_method: Literal["hyperfocal", "calibration_table"] = "hyperfocal"
    response_timeout: float = Field(gt=0, le=10, default=2.0)


class SensorsConfig(BaseModel):
    """Complete sensors configuration"""
    realsense: RealSenseConfig
    focus: FocusConfig

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Hardware Domain Root Validator
# -----------------------------------------------------------------------------

class HardwareConfig(BaseModel):
    """Complete hardware domain configuration"""
    camera: CameraConfig
    gimbal: GimbalConfig
    sensors: SensorsConfig

    model_config = {"extra": "forbid"}  # Hardware shouonly havld e these 3

# =============================================================================
# ALGORITHMS - VISION VALIDATORS
# =============================================================================

# -----------------------------------------------------------------------------
# YOLO Configuration Validators
# -----------------------------------------------------------------------------

class ModelSource(BaseModel):
    """YOLO model download and conversion settings"""
    base_model: str = Field(..., description="Base YOLO model name")
    download_url: Optional[str] = Field(default=None, description="Custom download URL")
    pt_model_path: str = Field(..., description="Path to PyTorch model file")
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
        """Validate base YOLO model name format"""
        import re
        pattern = r'^yolo(v)?(\d{1,2})(n|s|m|l|x)-(pose|seg|detect|obb|cls)$'
        if not re.match(pattern, v.lower()):
            raise ValueError(
                f"Invalid YOLO model name: {v}. "
                f"Expected format: yolo[v]<version><size>-<task> "
                f"e.g., yolo11n-pose, yolov8m-detect"
            )
        return v.lower()


class TensorRTConfig(BaseModel):
    """TensorRT engine settings"""
    precision: Literal["FP16", "FP32", "INT8"] = Field(default="FP16")
    batch_size: int = Field(default=1, ge=1, le=32)
    workspace_size: int = Field(default=4, ge=1, le=16)


class YOLOConfig(BaseModel):
    """YOLO-pose detection configuration"""
    model_path: str
    confidence_threshold: float = Field(ge=0.1, le=1.0, default=0.6)
    iou_threshold: float = Field(ge=0.1, le=1.0, default=0.45)
    max_detections: int = Field(ge=1, le=100, default=8)
    input_size: List[int] = Field(default=[640, 640])
    device: int = Field(ge=0, default=0)
    tensorrt: TensorRTConfig
    model_source: ModelSource

    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        if not v.endswith('.engine'):
            raise ValueError(f'Model must be .engine file, got: {v}')
        return v

    @field_validator('input_size')
    @classmethod
    def validate_input_size(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError(f'input_size must be [width, height]')
        if any(dim <= 0 or dim > 2048 for dim in v):
            raise ValueError(f'input_size must be 1-2048, got: {v}')
        return v


class FaceMeshConfig(BaseModel):
    """MediaPipe face mesh configuration"""
    max_num_faces: int = Field(ge=1, le=20, default=4)
    refine_landmarks: bool = True
    min_detection_confidence: float = Field(ge=0.1, le=1.0, default=0.6)
    min_tracking_confidence: float = Field(ge=0.1, le=1.0, default=0.5)
    static_image_mode: bool = False


class FaceDetectionConfig(BaseModel):
    """MediaPipe face detection configuration"""
    model_selection: Literal[0, 1] = 0
    min_detection_confidence: float = Field(ge=0.1, le=1.0, default=0.6)


class MediaPipeConfig(BaseModel):
    """MediaPipe face analysis configuration"""
    face_mesh: FaceMeshConfig = Field(default_factory=FaceMeshConfig)
    face_detection: FaceDetectionConfig = Field(default_factory=FaceDetectionConfig)


class VisionPerformanceConfig(BaseModel):
    """Vision performance benchmarks"""
    yolo_inference_ms: int = Field(ge=0, le=1000, default=15)
    face_analysis_ms: int = Field(ge=0, le=1000, default=10)
    depth_processing_ms: int = Field(ge=0, le=1000, default=5)


class VisionConfig(BaseModel):
    """Complete vision pipeline configuration"""
    yolo: YOLOConfig
    mediapipe: MediaPipeConfig = Field(default_factory=MediaPipeConfig)
    performance: VisionPerformanceConfig = Field(default_factory=VisionPerformanceConfig)

    model_config = {"extra": "forbid"}

# =============================================================================
# ALGORITHMS - POSITIONING & AUTO-FRAMING VALIDATORS
# =============================================================================

# -----------------------------------------------------------------------------
# Positioning Configuration Validators
# -----------------------------------------------------------------------------

class SceneDistanceRange(BaseModel):
    """Distance range for a scene type"""
    min_distance: float = Field(gt=0, le=15)
    max_distance: float = Field(gt=0, le=15)
    optimal_distance: float = Field(gt=0, le=15)
    max_horizontal_spread: Optional[float] = Field(default=None, gt=0, le=10)

    @model_validator(mode='after')
    def validate_distance_order(self) -> 'SceneDistanceRange':
        """Ensure distances are in logical order"""
        if self.min_distance >= self.max_distance:
            raise ValueError(f'min_distance ({self.min_distance}) must be < max_distance ({self.max_distance})')
        if not (self.min_distance <= self.optimal_distance <= self.max_distance):
            raise ValueError(f'optimal_distance ({self.optimal_distance}) must be between min and max')
        return self


class SceneClassificationConfig(BaseModel):
    """Scene classification distance ranges"""
    portrait: SceneDistanceRange
    couple: SceneDistanceRange
    small_group: SceneDistanceRange
    medium_group: SceneDistanceRange
    large_group: SceneDistanceRange


class CompositionPaddingScene(BaseModel):
    """Composition padding for a scene type"""
    horizontal: float = Field(ge=0.0, le=0.5, description="Horizontal padding percentage")
    vertical: float = Field(ge=0.0, le=0.5, description="Vertical padding percentage")


class CompositionPaddingConfig(BaseModel):
    """Composition padding for different scenes"""
    portrait: CompositionPaddingScene
    couple: CompositionPaddingScene
    small_group: CompositionPaddingScene
    medium_group: CompositionPaddingScene
    large_group: CompositionPaddingScene


class CoordinateSystemConfig(BaseModel):
    """Coordinate system configuration"""
    use_dynamic_transforms: bool = True
    transform_cache_duration: float = Field(gt=0, le=10, default=1.0)


class PositionFilteringConfig(BaseModel):
    """Position filtering settings"""
    enabled: bool = True
    median_window_size: int = Field(ge=1, le=20, default=5)
    max_position_change_per_frame: float = Field(gt=0, le=2.0, default=0.2)
    outlier_rejection_sigma: float = Field(gt=0, le=5.0, default=2.0)


class PositioningConfig(BaseModel):
    """Complete positioning configuration"""
    coordinate_system: CoordinateSystemConfig
    position_filtering: PositionFilteringConfig
    scene_classification: SceneClassificationConfig
    composition_padding: CompositionPaddingConfig
    max_calculation_time_ms: int = Field(ge=1, le=100, default=8)

    model_config = {"extra": "allow"}  # Allow additional fields


# -----------------------------------------------------------------------------
# Auto-Framing Core Configuration Validators
# -----------------------------------------------------------------------------

class FramingStrategyWeights(BaseModel):
    """Strategy weights for auto-framing"""
    composition: float = Field(ge=0.0, le=1.0)
    technical: float = Field(ge=0.0, le=1.0)
    aesthetic: float = Field(ge=0.0, le=1.0)
    uniqueness: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def weights_sum_to_one(self) -> 'FramingStrategyWeights':
        """Ensure all weights sum to approximately 1.0"""
        total = (self.composition + self.technical + self.aesthetic +
                 self.uniqueness + self.feasibility)
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f'Strategy weights must sum to 1.0, got {total:.4f}')
        return self


class AutoFramingCoreConfig(BaseModel):
    """Auto-framing core engine configuration"""
    enabled: bool = True
    default_mode: Literal["single_shot", "multi_shot", "creative_mode", "preset_sequence"] = "single_shot"
    default_priority: Literal[
        "composition_quality", "execution_speed", "subject_visibility", "creative_variety"] = "composition_quality"
    max_execution_time: float = Field(ge=1.0, le=60.0, default=20.0)
    quality_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    cache_size: int = Field(ge=1, le=1000, default=50)
    framing: Dict[str, Any]  # Contains nested configs

    model_config = {"extra": "allow"}


# -----------------------------------------------------------------------------
# Auto-Framing Composition Configuration Validators
# -----------------------------------------------------------------------------

class CompositionWeights(BaseModel):
    """Composition scoring weights"""
    rule_of_thirds: float = Field(ge=0.0, le=1.0)
    balance: float = Field(ge=0.0, le=1.0)
    subject_placement: float = Field(ge=0.0, le=1.0)
    negative_space: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def weights_sum_to_one(self) -> 'CompositionWeights':
        """Ensure composition weights sum to 1.0"""
        total = (self.rule_of_thirds + self.balance + self.subject_placement + self.negative_space)
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f'Composition weights must sum to 1.0, got {total:.4f}')
        return self


class CompositionThresholds(BaseModel):
    """Composition quality thresholds"""
    minimum_score: float = Field(ge=0.0, le=1.0, default=0.6)
    excellent_score: float = Field(ge=0.0, le=1.0, default=0.85)

    @model_validator(mode='after')
    def validate_threshold_order(self) -> 'CompositionThresholds':
        """Ensure minimum < excellent"""
        if self.minimum_score >= self.excellent_score:
            raise ValueError('minimum_score must be < excellent_score')
        return self


class RuleOfThirdsConfig(BaseModel):
    """Rule of thirds configuration"""
    tolerance_pixels: int = Field(ge=1, le=200, default=50)


class BalanceConfig(BaseModel):
    """Balance configuration"""
    optimal_deviation: float = Field(ge=0.0, le=0.5, default=0.15)


class EdgeSafetyConfig(BaseModel):
    """Edge safety configuration"""
    threshold: float = Field(ge=0.0, le=0.5, default=0.10)


class CircleDetectionConfig(BaseModel):
    """Circle detection settings"""
    enabled: bool = True
    min_radius: int = Field(ge=1, le=500, default=10)
    max_radius: int = Field(ge=1, le=1000, default=200)


class AnalyzerConfig(BaseModel):
    """Composition analyzer settings"""
    line_detection_threshold: int = Field(ge=1, le=255, default=100)
    min_line_length: int = Field(ge=1, le=500, default=50)
    max_line_gap: int = Field(ge=1, le=100, default=10)
    min_composition_score: float = Field(ge=0.0, le=1.0, default=0.3)
    target_composition_score: float = Field(ge=0.0, le=1.0, default=0.7)
    circle_detection: CircleDetectionConfig = Field(default_factory=CircleDetectionConfig)


class BackgroundAnalysisConfig(BaseModel):
    """Background analysis configuration"""
    edge_detection: Dict[str, int]
    color_analysis: Dict[str, Any]
    distraction_detection: Dict[str, int]
    thresholds: Dict[str, float]
    cache_size: int = Field(ge=1, le=1000, default=50)


class AutoFramingWeights(BaseModel):
    """Auto-framing specific composition weights"""
    rule_of_thirds: float = Field(ge=0.0, le=1.0)
    golden_ratio: float = Field(ge=0.0, le=1.0)
    subject_centering: float = Field(ge=0.0, le=1.0)
    leading_lines: float = Field(ge=0.0, le=1.0)
    symmetry: float = Field(ge=0.0, le=1.0)
    negative_space: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def weights_sum_to_one(self) -> 'AutoFramingWeights':
        """Ensure weights sum to 1.0"""
        total = (self.rule_of_thirds + self.golden_ratio + self.subject_centering + self.leading_lines + self.symmetry + self.negative_space)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f'Auto-framing weights must sum to 1.0, got {total:.4f}')
        return self


class QualityGates(BaseModel):
    """Quality gates for auto-framing"""
    minimum_acceptable: float = Field(ge=0.0, le=1.0, default=0.5)
    target_quality: float = Field(ge=0.0, le=1.0, default=0.75)
    excellent_quality: float = Field(ge=0.0, le=1.0, default=0.85)

    @model_validator(mode='after')
    def validate_quality_order(self) -> 'QualityGates':
        """Ensure quality gates are in order"""
        if not (self.minimum_acceptable <= self.target_quality <= self.excellent_quality):
            raise ValueError('Quality gates must be: minimum <= target <= excellent')
        return self


class AutoFramingCompositionConfig(BaseModel):
    """Auto-framing composition configuration"""
    enabled: bool = True
    weights: CompositionWeights
    thresholds: CompositionThresholds
    rule_of_thirds: RuleOfThirdsConfig = Field(default_factory=RuleOfThirdsConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)
    edge_safety: EdgeSafetyConfig = Field(default_factory=EdgeSafetyConfig)
    analyzer: AnalyzerConfig
    background: BackgroundAnalysisConfig
    auto_framing_weights: AutoFramingWeights
    quality_gates: QualityGates

    model_config = {"extra": "allow"}


# -----------------------------------------------------------------------------
# Auto-Framing Exposure Configuration Validators
# -----------------------------------------------------------------------------

class ApertureStrategy(BaseModel):
    """Aperture strategy for auto-framing"""
    preferred_aperture: float = Field(ge=1.4, le=22.0)
    fallback_apertures: List[float]
    reason: str

    @field_validator('fallback_apertures')
    @classmethod
    def validate_fallback_apertures(cls, v: List[float]) -> List[float]:
        """Validate fallback apertures are in valid range"""
        if not all(1.4 <= a <= 22.0 for a in v):
            raise ValueError('All fallback apertures must be between 1.4 and 22.0')
        return v


class FocusStrategy(BaseModel):
    """Focus strategy for auto-framing"""
    method: Literal["auto_canon", "manual_distance", "hyperfocal", "infinity"]
    fallback_method: str
    priority_points: Optional[str] = None
    hyperfocal_multiplier: Optional[float] = Field(default=None, ge=1.0, le=2.0)


class MotionCompensation(BaseModel):
    """Motion compensation for auto-framing movement"""
    movement_compensation_factor: float = Field(ge=1.0, le=3.0, default=1.5)
    min_shutter_auto_framing: Dict[str, str]
    use_ibis_during_movement: bool = True
    optical_is_priority: bool = True


class LightingThreshold(BaseModel):
    """Lighting threshold settings"""
    max_brightness: Optional[int] = Field(default=None, ge=0, le=255)
    min_brightness: Optional[int] = Field(default=None, ge=0, le=255)
    recommended_iso: int = Field(ge=100, le=102400)
    recommended_aperture: float = Field(ge=1.4, le=22.0)


class AutoFramingExposureConfig(BaseModel):
    """Auto-framing exposure configuration"""
    aperture_strategies: Dict[str, ApertureStrategy]
    focus_strategies: Dict[str, FocusStrategy]
    motion_compensation: MotionCompensation
    lighting_thresholds: Dict[str, LightingThreshold]

    model_config = {"extra": "allow"}


# -----------------------------------------------------------------------------
# Auto-Framing Root Validator
# -----------------------------------------------------------------------------

class AutoFramingConfig(BaseModel):
    """Complete auto-framing configuration"""
    core: AutoFramingCoreConfig
    composition: AutoFramingCompositionConfig
    exposure: AutoFramingExposureConfig

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Algorithms Domain Root Validator
# -----------------------------------------------------------------------------

class AlgorithmsConfig(BaseModel):
    """Complete algorithms domain configuration"""
    vision: VisionConfig
    positioning: PositioningConfig
    auto_framing: AutoFramingConfig

    model_config = {"extra": "forbid"}


# =============================================================================
# WORKFLOWS & SYSTEM VALIDATORS
# =============================================================================

# -----------------------------------------------------------------------------
# Workflows: Photo Capture Configuration Validators
# -----------------------------------------------------------------------------

class WorkflowTimingConfig(BaseModel):
    """Workflow timing settings"""
    max_session_time_s: float = Field(ge=1.0, le=300.0, default=60.0)
    max_positioning_time_s: float = Field(ge=1.0, le=120.0, default=30.0)
    capture_countdown_s: float = Field(ge=1.0, le=10.0, default=3.0)
    photos_per_session: int = Field(ge=1, le=10, default=1)


class ActivationConfig(BaseModel):
    """Workflow activation settings"""
    trigger_message: str = "y"
    auto_start: bool = False


class StateConfig(BaseModel):
    """State machine state configuration"""
    description: str
    timeout_s: Optional[float] = Field(default=None, ge=0, le=300)
    min_detection_time_s: Optional[float] = Field(default=None, ge=0, le=30)
    min_person_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_guidance_attempts: Optional[int] = Field(default=None, ge=1, le=20)
    position_tolerance_m: Optional[float] = Field(default=None, ge=0, le=2.0)
    required_checks: Optional[List[str]] = None
    duration_s: Optional[float] = Field(default=None, ge=0, le=30)
    display_time_s: Optional[float] = Field(default=None, ge=0, le=30)
    actions: Optional[List[str]] = None

    model_config = {"extra": "allow"}


class VoiceTimingConfig(BaseModel):
    """Voice guidance timing"""
    message_interval_s: float = Field(ge=0.5, le=10.0, default=2.0)
    position_update_interval_s: float = Field(ge=0.5, le=10.0, default=3.0)


class VoiceGuidanceConfig(BaseModel):
    """Voice guidance configuration"""
    language: str = "en"
    timing: VoiceTimingConfig
    messages: Dict[str, Any]  # Contains message lists


class QualityChecksConfig(BaseModel):
    """Quality check settings"""
    min_face_size_pixels: int = Field(ge=10, le=500, default=80)
    max_motion_blur_threshold: float = Field(ge=0.0, le=50.0, default=5.0)
    min_contrast_level: float = Field(ge=0.0, le=1.0, default=0.3)


class SubjectPositioningConfig(BaseModel):
    """Subject positioning preferences"""
    vertical_center_offset: float = Field(ge=-0.5, le=0.5, default=0.1)
    horizontal_tolerance: float = Field(ge=0.0, le=0.5, default=0.05)


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration"""
    depth_failures: Dict[str, Any]
    detection_failures: Dict[str, Any]
    control_failures: Dict[str, Any]
    system_failures: Dict[str, Any]


class AutoFramingWorkflowTimingConfig(BaseModel):
    """Auto-framing workflow timing"""
    max_framing_time: float = Field(ge=1.0, le=60.0, default=15.0)
    positioning_timeout: float = Field(ge=1.0, le=30.0, default=10.0)
    quality_improvement_timeout: float = Field(ge=1.0, le=30.0, default=5.0)
    step_delays: Dict[str, float]

    @field_validator('step_delays')
    @classmethod
    def validate_step_delays(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate step delays are reasonable"""
        for key, value in v.items():
            if not 0.1 <= value <= 10.0:
                raise ValueError(f'step_delays.{key} must be between 0.1 and 10.0 seconds')
        return v


class AutoFramingWorkflowActivationConfig(BaseModel):
    """Auto-framing activation settings"""
    auto_on_detection: bool = True
    manual_trigger_topic: str = Field(pattern=r'^/[\w/]+$')


class AutoFramingWorkflowExecutionConfig(BaseModel):
    """Auto-framing execution modes"""
    single_optimization: bool = True
    continuous_tracking: bool = False
    multi_shot_sequence: bool = False


class AutoFramingWorkflowConfig(BaseModel):
    """Auto-framing workflow integration"""
    enabled: bool = True
    activation: AutoFramingWorkflowActivationConfig
    timing: AutoFramingWorkflowTimingConfig
    execution_modes: AutoFramingWorkflowExecutionConfig


class PhotoCaptureWorkflowConfig(BaseModel):
    """Complete photo capture workflow configuration"""
    workflow: WorkflowTimingConfig
    activation: ActivationConfig
    states: Dict[str, StateConfig]
    voice_guidance: VoiceGuidanceConfig
    quality_checks: QualityChecksConfig
    subject_positioning: SubjectPositioningConfig
    error_handling: ErrorHandlingConfig
    auto_framing_workflow: AutoFramingWorkflowConfig

    @field_validator('states')
    @classmethod
    def validate_required_states(cls, v: Dict[str, StateConfig]) -> Dict[str, StateConfig]:
        """Ensure all required states are present"""
        required_states = [
            'idle', 'initializing', 'detecting', 'positioning',
            'adjusting_camera', 'verifying', 'countdown', 'capturing', 'complete'
        ]
        missing = set(required_states) - set(v.keys())
        if missing:
            raise ValueError(f'Missing required states: {missing}')
        return v


class WorkflowsConfig(BaseModel):
    """Complete workflows domain configuration"""
    photo_capture: PhotoCaptureWorkflowConfig

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# System: Performance Configuration Validators
# -----------------------------------------------------------------------------

class ComponentPerformanceConfig(BaseModel):
    """Component-specific performance settings"""
    depth_processing: Dict[str, Any]
    position_calculation: Dict[str, Any]
    control_system: Dict[str, Any]


class ControlTimingConfig(BaseModel):
    """Control loop timing"""
    main_loop_hz: float = Field(ge=1.0, le=100.0, default=5.0)
    vision_processing_hz: float = Field(ge=1.0, le=100.0, default=10.0)
    depth_processing_hz: float = Field(ge=1.0, le=100.0, default=5.0)
    position_update_hz: float = Field(ge=1.0, le=100.0, default=5.0)


class ThreadingConfig(BaseModel):
    """Threading configuration"""
    max_workers: int = Field(ge=1, le=32, default=4)
    thread_timeout: float = Field(gt=0, le=300, default=30.0)
    use_async: bool = True


class TestingConfig(BaseModel):
    """Testing configuration"""
    test_image_path: str
    test_depth_path: str
    mock_canon_server: bool = True
    performance_benchmarks: Dict[str, int]


class PerformanceConfig(BaseModel):
    """Complete performance configuration"""
    components: ComponentPerformanceConfig
    control_timing: ControlTimingConfig
    threading: ThreadingConfig
    testing: TestingConfig

    model_config = {"extra": "allow"}


# -----------------------------------------------------------------------------
# System: Logging Configuration Validators
# -----------------------------------------------------------------------------

class LoggingFeaturesConfig(BaseModel):
    """Logging features configuration"""
    exposure_calculations: bool = True
    composition_scores: bool = True
    lighting_analysis: bool = True
    calculation_time: bool = True
    save_debug_frames: bool = False
    debug_output_path: str = "/tmp/manriix_debug"


class LoggingConfig(BaseModel):
    """Complete logging configuration"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["structured", "simple"] = "structured"
    file_path: str = "logs/manriix_photo_va.log"
    max_file_size_mb: int = Field(ge=1, le=1000, default=100)
    backup_count: int = Field(ge=1, le=10, default=5)
    console_output: bool = True
    features: LoggingFeaturesConfig

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# System: ROS2 Configuration Validators
# -----------------------------------------------------------------------------

class QoSConfig(BaseModel):
    """ROS2 QoS settings"""
    reliability: Literal["reliable", "best_effort"] = "reliable"
    durability: Literal["volatile", "transient_local"] = "volatile"
    history: Literal["keep_last", "keep_all"] = "keep_last"
    depth: int = Field(ge=1, le=100, default=10)


class NodeParamsConfig(BaseModel):
    """ROS2 node parameters"""
    use_sim_time: bool = False
    publish_rate_hz: float = Field(ge=1.0, le=100.0, default=10.0)


class TFConfig(BaseModel):
    """ROS2 TF settings"""
    publish_tf: bool = True
    tf_prefix: str = ""
    update_rate_hz: float = Field(ge=1.0, le=100.0, default=20.0)


class ROS2Config(BaseModel):
    """Complete ROS2 configuration"""
    qos: QoSConfig
    node_params: NodeParamsConfig
    tf: TFConfig

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# System Domain Root Validator
# -----------------------------------------------------------------------------

class SystemConfig(BaseModel):
    """Complete system domain configuration"""
    performance: PerformanceConfig
    logging: LoggingConfig
    ros2: ROS2Config

    model_config = {"extra": "forbid"}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_hardware_config(config: Dict[str, Any]) -> HardwareConfig:
    """
    Validate hardware configuration with comprehensive checks

    Args:
        config: Hardware configuration dictionary

    Returns:
        Validated HardwareConfig model

    Raises:
        ValidationError: If configuration is invalid
    """
    return HardwareConfig(**config)


def validate_vision_config(config: Dict[str, Any]) -> VisionConfig:
    """
    Validate vision configuration with comprehensive checks

    Args:
        config: Vision configuration dictionary

    Returns:
        Validated VisionConfig model

    Raises:
        ValidationError: If configuration is invalid
    """
    return VisionConfig(**config)


def validate_positioning_config(config: Dict[str, Any]) -> PositioningConfig:
    """Validate positioning configuration"""
    return PositioningConfig(**config)


def validate_auto_framing_config(config: Dict[str, Any]) -> AutoFramingConfig:
    """Validate auto-framing configuration"""
    return AutoFramingConfig(**config)


def validate_algorithms_config(config: Dict[str, Any]) -> AlgorithmsConfig:
    """
    Validate complete algorithms domain configuration

    Args:
        config: Algorithms configuration dictionary

    Returns:
        Validated AlgorithmsConfig model

    Raises:
        ValidationError: If configuration is invalid
    """
    return AlgorithmsConfig(**config)


def validate_workflows_config(config: Dict[str, Any]) -> WorkflowsConfig:
    """
    Validate workflows configuration

    Args:
        config: Workflows configuration dictionary

    Returns:
        Validated WorkflowsConfig model

    Raises:
        ValidationError: If configuration is invalid
    """
    return WorkflowsConfig(**config)


def validate_system_config(config: Dict[str, Any]) -> SystemConfig:
    """
    Validate system configuration

    Args:
        config: System configuration dictionary

    Returns:
        Validated SystemConfig model

    Raises:
        ValidationError: If configuration is invalid
    """
    return SystemConfig(**config)


# =============================================================================
# PART 5: INTEGRATION FUNCTIONS & COMPLETE VALIDATION
# =============================================================================

def validate_all_domains(
        hardware: Dict[str, Any],
        algorithms: Dict[str, Any],
        workflows: Dict[str, Any],
        system: Dict[str, Any]
) -> Tuple[HardwareConfig, AlgorithmsConfig, WorkflowsConfig, SystemConfig]:
    """
    Validate all domain configurations at once

    Args:
        hardware: Hardware domain configuration dictionary
        algorithms: Algorithms domain configuration dictionary
        workflows: Workflows domain configuration dictionary
        system: System domain configuration dictionary

    Returns:
        Tuple of validated models: (HardwareConfig, AlgorithmsConfig, WorkflowsConfig, SystemConfig)

    Raises:
        ValidationError: If any configuration is invalid
    """
    hw = validate_hardware_config(hardware)
    alg = validate_algorithms_config(algorithms)
    wf = validate_workflows_config(workflows)
    sys_config = validate_system_config(system)

    return hw, alg, wf, sys_config


def validate_domain_consistency(
        hardware: HardwareConfig,
        algorithms: AlgorithmsConfig,
        workflows: WorkflowsConfig,
        system: SystemConfig
) -> bool:
    """
    Check cross-domain consistency

    Validates:
    - Vision Hz >= Main loop Hz
    - Auto-framing time < Session time
    - Gimbal velocity compatibility

    Args:
        hardware: Validated hardware config
        algorithms: Validated algorithms config
        workflows: Validated workflows config
        system: Validated system config

    Returns:
        True if consistent

    Raises:
        ValueError: If cross-domain inconsistencies found
    """
    # Check control timing compatibility
    main_loop_hz = system.performance.control_timing.main_loop_hz
    vision_hz = system.performance.control_timing.vision_processing_hz

    if vision_hz < main_loop_hz:
        raise ValueError(
            f'Vision Hz ({vision_hz}) should be >= main loop Hz ({main_loop_hz})'
        )

    # Check auto-framing timing
    af_workflow = workflows.photo_capture.auto_framing_workflow
    if af_workflow.enabled:
        max_framing = af_workflow.timing.max_framing_time
        max_session = workflows.photo_capture.workflow.max_session_time_s

        if max_framing > max_session:
            raise ValueError(
                f'Auto-framing time ({max_framing}s) > session time ({max_session}s)'
            )

    return True


def get_validation_summary(
        hardware: HardwareConfig,
        algorithms: AlgorithmsConfig,
        workflows: WorkflowsConfig,
        system: SystemConfig
) -> str:
    """Get formatted validation summary"""
    return f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    CONFIGURATION VALIDATION SUMMARY                       ║
╚══════════════════════════════════════════════════════════════════════════╝

✅ HARDWARE
   Camera: {'Manual' if hardware.camera.exposure_control.enabled else 'Auto'} | ISO {hardware.camera.iso_preferences.preferred_base}
   Gimbal: {hardware.gimbal.gimbal_control.control_method} | {hardware.gimbal.motion_constraints.max_pan_velocity}°/s
   Focus: {hardware.sensors.focus.calculation_method}

✅ ALGORITHMS
   YOLO: {algorithms.vision.yolo.confidence_threshold} conf | {algorithms.vision.yolo.model_source.base_model}
   Scenes: {len(algorithms.positioning.scene_classification.__dict__)} types
   Auto-framing: {'Enabled' if algorithms.auto_framing.core.enabled else 'Disabled'} | Q={algorithms.auto_framing.core.quality_threshold}

✅ WORKFLOWS
   Session: {workflows.photo_capture.workflow.max_session_time_s}s | States: {len(workflows.photo_capture.states)}
   Auto-framing: {'On' if workflows.photo_capture.auto_framing_workflow.enabled else 'Off'}

✅ SYSTEM
   Timing: {system.performance.control_timing.main_loop_hz}Hz | Workers: {system.performance.threading.max_workers}
   Logging: {system.logging.level} | ROS2: {system.ros2.qos.reliability}

╚══════════════════════════════════════════════════════════════════════════╝
"""


# Quick boolean validators (no exceptions)
def quick_validate_hardware(config: Dict[str, Any]) -> bool:
    try:
        validate_hardware_config(config)
        return True
    except:
        return False


def quick_validate_algorithms(config: Dict[str, Any]) -> bool:
    try:
        validate_algorithms_config(config)
        return True
    except:
        return False


def quick_validate_workflows(config: Dict[str, Any]) -> bool:
    try:
        validate_workflows_config(config)
        return True
    except:
        return False


def quick_validate_system(config: Dict[str, Any]) -> bool:
    try:
        validate_system_config(config)
        return True
    except:
        return False


# Export list for easy imports
__all__ = [
    # Root configs
    'HardwareConfig',
    'AlgorithmsConfig',
    'WorkflowsConfig',
    'SystemConfig',

    # Validation functions
    'validate_hardware_config',
    'validate_vision_config',
    'validate_positioning_config',
    'validate_auto_framing_config',
    'validate_algorithms_config',
    'validate_workflows_config',
    'validate_system_config',
    'validate_all_domains',
    'validate_domain_consistency',

    # Utilities
    'get_validation_summary',
    'quick_validate_hardware',
    'quick_validate_algorithms',
    'quick_validate_workflows',
    'quick_validate_system',
]




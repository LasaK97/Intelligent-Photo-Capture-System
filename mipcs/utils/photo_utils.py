import numpy as np
import cv2
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import yaml
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_exposure_config, reload_exposure_config, ExposureConfig, PhotoPreferences
from ..utils.logger import get_logger

logger = get_logger(__name__)

class PhotoUtilsConfig:

    def __init__(self):

        exposure_config = get_exposure_config()

        #extract specs
        # camera
        sensor = exposure_config.camera_specs.sensor
        lens = exposure_config.camera_specs.lens
        iso_specs = exposure_config.camera_specs.iso

        # sensor
        self.sensor_width_mm = sensor.width_mm
        self.sensor_height_mm = sensor.height_mm
        self.crop_factor = sensor.crop_factor
        self.circle_of_confusion_mm = sensor.circle_of_confusion_mm

        # lens
        self.focal_length_min = lens.focal_length_min
        self.focal_length_max = lens.focal_length_max
        self.f_number_min = lens.f_number_min
        self.f_number_max = lens.f_number_max
        self.available_apertures = lens.available_apertures

        #legacy aliases --> backward compatibility
        self.aperture_min = self.f_number_min
        self.aperture_max = self.f_number_max

        # ISO specs
        self.iso_min = iso_specs.min
        self.iso_max = iso_specs.max
        self.iso_standard_values = iso_specs.standard_values

        #exposure control settings
        self.exposure_enabled = exposure_config.exposure_control.enabled
        self.log_recommendations = exposure_config.exposure_control.log_recommendations

        #preferences
        self.preferences = self._convert_preferences_to_dict(exposure_config.preferences)
        self.iso_acceptable_max = self.preferences['iso']['acceptable_max']

        # lightning
        self.lighting_thresholds = self._convert_lighting_thresholds(
                    exposure_config.lighting_analysis.thresholds
                )

        # histogram thresholds
        self.histogram_contrast_threshold = exposure_config.lighting_analysis.histogram.contrast_threshold

        # composition enabled
        self.composition_enabled = exposure_config.composition.enabled
        self.composition_weights = {
            'rule_of_thirds': exposure_config.composition.weights.rule_of_thirds,
            'balance': exposure_config.composition.weights.balance,
            'subject_placement': exposure_config.composition.weights.subject_placement,
            'negative_space': exposure_config.composition.weights.negative_space
        }

        # validation settings
        self.strict_mode = exposure_config.validation.strict_mode
        self.auto_correct = exposure_config.validation.auto_correct

    def _convert_preferences_to_dict(self, prefs) -> Dict[str, Any]:
        """convert preferences to dict"""
        return {
            'aperture_ranges': {
                'portrait_single': {
                    'min': prefs.aperture_ranges.portrait_single.min,
                    'max': prefs.aperture_ranges.portrait_single.max,
                    'preferred': prefs.aperture_ranges.portrait_single.preferred,
                    'reason': prefs.aperture_ranges.portrait_single.reason
                },
                'portrait_couple': {
                    'min': prefs.aperture_ranges.portrait_couple.min,
                    'max': prefs.aperture_ranges.portrait_couple.max,
                    'preferred': prefs.aperture_ranges.portrait_couple.preferred,
                    'reason': prefs.aperture_ranges.portrait_couple.reason
                },
                'group_small': {
                    'min': prefs.aperture_ranges.group_small.min,
                    'max': prefs.aperture_ranges.group_small.max,
                    'preferred': prefs.aperture_ranges.group_small.preferred,
                    'reason': prefs.aperture_ranges.group_small.reason
                },
                'group_large': {
                    'min': prefs.aperture_ranges.group_large.min,
                    'max': prefs.aperture_ranges.group_large.max,
                    'preferred': prefs.aperture_ranges.group_large.preferred,
                    'reason': prefs.aperture_ranges.group_large.reason
                },
                'landscape': {
                    'min': prefs.aperture_ranges.landscape.min,
                    'max': prefs.aperture_ranges.landscape.max,
                    'preferred': prefs.aperture_ranges.landscape.preferred,
                    'reason': prefs.aperture_ranges.landscape.reason
                },
                'action': {
                    'min': prefs.aperture_ranges.action.min,
                    'max': prefs.aperture_ranges.action.max,
                    'preferred': prefs.aperture_ranges.action.preferred,
                    'reason': prefs.aperture_ranges.action.reason
                }
            },
            'shutter_speed': {
                'min_handheld': prefs.shutter_speed.min_handheld,
                'min_stabilized': prefs.shutter_speed.min_stabilized,
                'portrait_min': prefs.shutter_speed.portrait_min,
                'portrait_close_min': prefs.shutter_speed.portrait_close_min,
                'action_min': prefs.shutter_speed.action_min,
                'sports_min': prefs.shutter_speed.sports_min,
                'focal_length_rule_multiplier': prefs.shutter_speed.focal_length_rule_multiplier
            },
            'iso': {
                'preferred_base': prefs.iso.preferred_base,
                'acceptable_max': prefs.iso.acceptable_max,
                'emergency_max': prefs.iso.emergency_max,
                'portrait_max': prefs.iso.portrait_max,
                'landscape_max': prefs.iso.landscape_max,
                'action_max': prefs.iso.action_max
            },
            'distance_thresholds': {
                'close_portrait': prefs.distance_thresholds.close_portrait,
                'standard_portrait': prefs.distance_thresholds.standard_portrait,
                'group_distance': prefs.distance_thresholds.group_distance,
                'far_distance': prefs.distance_thresholds.far_distance
            }
        }

    def _convert_lighting_thresholds(self, thresholds) -> Dict[str, Any]:
        """convert lighting to dict"""
        return {
            'very_dark': {
                'max': thresholds.very_dark.max,
                'recommended_iso': thresholds.very_dark.recommended_iso,
                'recommended_aperture': thresholds.very_dark.recommended_aperture
            },
            'dark': {
                'min': thresholds.dark.min,
                'max': thresholds.dark.max,
                'recommended_iso': thresholds.dark.recommended_iso,
                'recommended_aperture': thresholds.dark.recommended_aperture
            },
            'moderate': {
                'min': thresholds.moderate.min,
                'max': thresholds.moderate.max,
                'recommended_iso': thresholds.moderate.recommended_iso,
                'recommended_aperture': thresholds.moderate.recommended_aperture
            },
            'bright': {
                'min': thresholds.bright.min,
                'max': thresholds.bright.max,
                'recommended_iso': thresholds.bright.recommended_iso,
                'recommended_aperture': thresholds.bright.recommended_aperture
            },
            'very_bright': {
                'min': thresholds.very_bright.min,
                'recommended_iso': thresholds.very_bright.recommended_iso,
                'recommended_aperture': thresholds.very_bright.recommended_aperture
            }
        }

#================================================================================================
try:
    _config = PhotoUtilsConfig()
except Exception as e:
    logger.error(f"Failed to load exposure config: {e}")
    raise RuntimeError(f"Cannot initialize PhotoUtils without valid config: {e}")

SENSOR_WIDTH_MM = _config.sensor_width_mm
SENSOR_HEIGHT_MM = _config.sensor_height_mm
CROP_FACTOR = _config.crop_factor
CIRCLE_OF_CONFUSION_MM = _config.circle_of_confusion_mm
FOCAL_LENGTH_MIN = _config.focal_length_min
FOCAL_LENGTH_MAX = _config.focal_length_max
APERTURE_MIN = _config.aperture_min  # Minimum f-number (widest aperture)
APERTURE_MAX = _config.aperture_max  # Maximum f-number (smallest aperture)
ISO_MIN = _config.iso_min
ISO_MAX = _config.iso_max
ISO_ACCEPTABLE_MAX = _config.iso_acceptable_max


class LightingCondition(Enum):
    """Lighting conditions based on frame analysis"""
    VERY_DARK = "very_dark"      # < 50 brightness
    DARK = "dark"                # 50-100
    MODERATE = "moderate"        # 100-150
    BRIGHT = "bright"            # 150-200
    VERY_BRIGHT = "very_bright"  # > 200


class SceneType(Enum):
    """Scene types for exposure calculation"""
    PORTRAIT_SINGLE = "portrait_single"
    PORTRAIT_COUPLE = "portrait_couple"
    GROUP_SMALL = "group_small"      # 3-5 people
    GROUP_LARGE = "group_large"      # 6+ people
    LANDSCAPE = "landscape"
    ACTION = "action"
    GENERAL = "general"


@dataclass
class ExposureSettings:
    """Calculated exposure settings"""
    aperture: float  # f-stop
    shutter_speed: str  # e.g., "1/125"
    iso: int
    lighting: LightingCondition
    scene_type: SceneType
    reasoning: str
    confidence: float  # 0-1


@dataclass
class DepthOfFieldResult:
    """Depth of field calculation result"""
    near_limit: float  # meters
    far_limit: float  # meters
    total_dof: float  # meters
    hyperfocal_distance: float  # meters
    in_focus_range: Tuple[float, float]  # (near, far) in meters


@dataclass
class CompositionScore:
    """Composition analysis result"""
    overall_score: float  # 0-1
    rule_of_thirds_score: float
    balance_score: float
    subject_placement_score: float
    negative_space_score: float
    recommendations: List[str]

def estimate_lighting_from_frame(
        frame: np.ndarray,
        config: Optional[PhotoUtilsConfig] = None
) -> Dict[str, Any]:
    """Estimate lightning from frame
    USE brightness histogram and stat
    """

    if config is None:
        config = _config

    # convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame


    # cal brightness metrics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    median_brightness = np.median(gray)

    # cal histogram
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # get thresholds
    thresholds = config.lighting_thresholds

    # classify lightning condition
    if mean_brightness < thresholds['very_dark']['max']:
        condition = LightingCondition.VERY_DARK
        recommended_iso = thresholds['very_dark']['recommended_iso']
        recommended_aperture = thresholds['very_dark']['recommended_aperture']
    elif mean_brightness < thresholds['dark']['max']:
        condition = LightingCondition.DARK
        recommended_iso = thresholds['dark']['recommended_iso']
        recommended_aperture = thresholds['dark']['recommended_aperture']
    elif mean_brightness < thresholds['moderate']['max']:
        condition = LightingCondition.MODERATE
        recommended_iso = thresholds['moderate']['recommended_iso']
        recommended_aperture = thresholds['moderate']['recommended_aperture']
    elif mean_brightness < thresholds['bright']['max']:
        condition = LightingCondition.BRIGHT
        recommended_iso = thresholds['bright']['recommended_iso']
        recommended_aperture = thresholds['bright']['recommended_aperture']
    else:
        condition = LightingCondition.VERY_BRIGHT
        recommended_iso = thresholds['very_bright']['recommended_iso']
        recommended_aperture = thresholds['very_bright']['recommended_aperture']

    # detect if histogram is well distributed (good contrast)

    hist_spread = np.std(histogram)
    has_good_contrast = std_brightness < config.histogram_contrast_threshold

    return {
        'condition': condition,
        'mean_brightness': float(mean_brightness),
        'median_brightness': float(median_brightness),
        'std_brightness': float(std_brightness),
        'histogram': histogram,
        'has_good_contrast': has_good_contrast,
        'histogram_spread': float(hist_spread),
        'recommended_iso_base': recommended_iso,
        'recommended_aperture_base': recommended_aperture
    }

def classify_scene_type(
        subject_count: int,
        scene_context: str = "general"
) -> SceneType:
    """Classify scene type"""
    if scene_context == "action":
        return SceneType.ACTION
    elif scene_context == "landscape":
        return SceneType.LANDSCAPE

        # Person-based classification
    if subject_count == 1:
        return SceneType.PORTRAIT_SINGLE
    elif subject_count == 2:
        return SceneType.PORTRAIT_COUPLE
    elif 3 <= subject_count <= 5:
        return SceneType.GROUP_SMALL
    elif subject_count >= 6:
        return SceneType.GROUP_LARGE
    else:
        return SceneType.GENERAL

def calculate_optimal_exposure(
frame: np.ndarray,
    subject_count: int,
    subject_distance: float,
    scene_context: str = "general",
    focal_length: float = 50.0,
    config: Optional[PhotoUtilsConfig] = None,
    preferences: Optional[Dict[str, Any]] = None
)-> ExposureSettings:
    """Calculate optimal exposure based on scene analysis"""

    if config is None:
        config = _config

    # check if exposure control is disabled
    if not config.exposure_enabled:
        logger.info("Exposure control disabled - using camera Auto mode")
        return ExposureSettings(
            aperture=config.aperture_min,
            shutter_speed="1/125",
            iso=400,
            lighting=LightingCondition.MODERATE,
            scene_type=SceneType.GENERAL,
            reasoning="Exposure control disabled - camera in Auto mode",
            confidence=1.0
        )

    #get preferences
    if preferences is None:
        preferences = config.preferences

    # analysis lightning
    lighting_analysis = estimate_lighting_from_frame(frame, config)
    lighting_condition = lighting_analysis['condition']

    # classify scene
    scene_type = classify_scene_type(subject_count, scene_context)

    # 1. Determine aperture

    aperture_ranges = preferences['aperture_ranges']

    if scene_type == SceneType.PORTRAIT_SINGLE:
        ap_cfg = aperture_ranges['portrait_single']
        aperture = ap_cfg['preferred']
        reasoning = ap_cfg['reason']
    elif scene_type == SceneType.PORTRAIT_COUPLE:
        ap_cfg = aperture_ranges['portrait_couple']
        aperture = ap_cfg['preferred']
        reasoning = ap_cfg['reason']
    elif scene_type == SceneType.GROUP_SMALL:
        ap_cfg = aperture_ranges['group_small']
        aperture = ap_cfg['preferred']
        reasoning = ap_cfg['reason']
    elif scene_type == SceneType.GROUP_LARGE:
        ap_cfg = aperture_ranges['group_large']
        aperture = ap_cfg['preferred']
        reasoning = ap_cfg['reason']
    elif scene_type == SceneType.LANDSCAPE:
        ap_cfg = aperture_ranges['landscape']
        aperture = ap_cfg['preferred']
        reasoning = ap_cfg['reason']
    elif scene_type == SceneType.ACTION:
        ap_cfg = aperture_ranges['action']
        aperture = ap_cfg['preferred']
        reasoning = ap_cfg['reason']
    else:
        aperture = lighting_analysis['recommended_aperture_base']
        reasoning = f"General scene: aperture based on {lighting_condition.value} lighting"

    # clamp lens  capabilities
    aperture = max(config.aperture_min, min(config.aperture_max, aperture))

    # auto correct nearest available apeture
    if config.auto_correct and config.available_apertures:
        aperture = min(config.available_apertures, key=lambda x: abs(x - aperture))

    # 2. determine shutter speed

    shutter_prefs = preferences['shutter_speed']

    if scene_type == SceneType.ACTION:
        shutter_speed = shutter_prefs['action_min']
    elif subject_distance < 3.0:  # Close portraits
        shutter_speed = shutter_prefs['portrait_close_min']
    else:
        # Use 1/(focal_length × multiplier) rule with IBIS
        multiplier = shutter_prefs['focal_length_rule_multiplier']
        min_shutter = int(focal_length * multiplier)
        min_handheld = shutter_prefs['min_handheld']
        min_handheld_value = int(min_handheld.split('/')[1]) if '/' in min_handheld else 125

        shutter_value = max(min_shutter, min_handheld_value)
        shutter_speed = f"1/{shutter_value}"

    # 3. Cal ISO
    iso = lighting_analysis['recommended_iso_base']

    # adjust for aperture
    if aperture > 5.6:
        iso = int(iso * 1.5)
    elif aperture > 8.0:
        iso = int(iso * 2)

    # adjust for shutter speed
    shutter_value = int(shutter_speed.split('/')[1]) if '/' in shutter_speed else 1
    if shutter_value >= 250:
        iso = int(iso * 1.3)

    # clamp to acceptable range
    iso = min(iso, preferences['iso']['acceptable_max'])
    iso = max(config.iso_min, iso)

    # round to nearest standard ISO value
    if config.iso_standard_values:
        iso = min(config.iso_standard_values, key=lambda x: abs(x - iso))

    # 4. validate settings
    if config.strict_mode:
        is_valid, warnings = validate_exposure_settings_internal(aperture, iso, config)
        if not is_valid:
            raise ValueError(f"Invalid exposure settings: {warnings}")

        # calculate confidence
    if lighting_condition in [LightingCondition.VERY_DARK, LightingCondition.VERY_BRIGHT]:
        confidence = 0.7
    elif lighting_condition in [LightingCondition.DARK, LightingCondition.BRIGHT]:
        confidence = 0.85
    else:
        confidence = 0.95

    settings = ExposureSettings(
        aperture=aperture,
        shutter_speed=shutter_speed,
        iso=iso,
        lighting=lighting_condition,
        scene_type=scene_type,
        reasoning=reasoning,
        confidence=confidence
    )

    if config.log_recommendations:
        logger.info(f"Calculated exposure: {format_exposure_for_display(settings)}")

    return settings


def validate_exposure_settings_internal(
        aperture: float,
        iso: float,
        config: PhotoUtilsConfig
) -> Tuple[bool, List[str]]:
    """internal validation helper"""

    warnings = []
    is_valid = True

    if aperture < config.aperture_min or aperture > config.aperture_max:
        warnings.append(
            f"Aperture f/{aperture} out of range [{config.aperture_min}-{config.aperture_max}]"
        )
        is_valid = False

    if iso < config.iso_min or iso > config.iso_max:
        warnings.append(f"ISO {iso} out of range [{config.iso_min}-{config.iso_max}]")
        is_valid = False

    if iso > config.iso_acceptable_max:
        warnings.append(f"ISO {iso} exceeds quality threshold ({config.iso_acceptable_max})")

    return is_valid, warnings

def calculate_depth_of_field(
        focal_length: float,
        aperture: float,
        subject_distance: float,
        coc: float = CIRCLE_OF_CONFUSION_MM
) -> DepthOfFieldResult:
    """Calculate depth of field for given camera settings"""
    subject_distance_mm = subject_distance * 1000
    hyperfocal = (focal_length ** 2) / (aperture * coc) + focal_length

    if subject_distance_mm < hyperfocal:
        near_limit_mm = (subject_distance_mm * (hyperfocal - focal_length)) /  (hyperfocal + subject_distance_mm - 2 * focal_length)
        far_limit_mm = (subject_distance_mm * (hyperfocal - focal_length)) / (hyperfocal - subject_distance_mm)
    else:
        near_limit_mm = hyperfocal / 2
        far_limit_mm = float('inf')

    near_limit = near_limit_mm / 1000
    far_limit = far_limit_mm / 1000 if far_limit_mm != float('inf') else float('inf')
    hyperfocal_m = hyperfocal / 1000

    total_dof = float('inf') if far_limit == float('inf') else far_limit - near_limit

    return DepthOfFieldResult(
        near_limit=near_limit,
        far_limit=far_limit,
        total_dof=total_dof,
        hyperfocal_distance=hyperfocal_m,
        in_focus_range=(near_limit, far_limit)
    )

def calculate_hyperfocal_distance(
        focal_length: float,
        aperture: float,
        coc: float = CIRCLE_OF_CONFUSION_MM
) -> float:
    """Calculate hyperfocal distance for maximum depth of field"""

    hyperfocal_mm = (focal_length ** 2) / (aperture * coc) + focal_length
    return hyperfocal_mm / 1000


def calculate_optimal_aperture_for_dof(
    focal_length: float,
    subject_distance: float,
    desired_dof: float,
    coc: float = CIRCLE_OF_CONFUSION_MM
) -> float:
    """calculate aperture needed to achieve desired depth of field"""
    s_mm = subject_distance * 1000
    dof_mm = desired_dof * 1000

    aperture = (dof_mm * focal_length ** 2) / (2 * coc * s_mm ** 2)
    aperture = max(APERTURE_MIN, min(APERTURE_MAX, aperture))

    standard_apertures = [2.8, 3.2, 3.5, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 9.0, 10.0, 11.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0]
    aperture = min(standard_apertures, key=lambda x: abs(x - aperture))

    return aperture


## FOV cal
def calculate_field_of_view(focal_length: float, sensor_dimension: float) -> float:
    """calculate field of view angle for given focal length."""
    fov_radians = 2 * np.arctan(sensor_dimension / (2 * focal_length))
    return np.degrees(fov_radians)


def calculate_horizontal_fov(focal_length: float) -> float:
    """calculate horizontal field of view for full-frame sensor"""
    return calculate_field_of_view(focal_length, SENSOR_WIDTH_MM)


def calculate_vertical_fov(focal_length: float) -> float:
    """calculate vertical field of view for full-frame sensor"""
    return calculate_field_of_view(focal_length, SENSOR_HEIGHT_MM)


def calculate_subject_size_in_frame(
    subject_height: float,
    subject_distance: float,
    focal_length: float
) -> float:
    """calculate what percentage of frame height the subject will occupy."""
    vfov_rad = 2 * np.arctan(SENSOR_HEIGHT_MM / (2 * focal_length))
    frame_height_at_distance = 2 * subject_distance * np.tan(vfov_rad / 2)
    percentage = subject_height / frame_height_at_distance
    return min(1.0, percentage)

# composition analysis

def calculate_composition_score(
        frame: np.ndarray,
        subject_positions: List[Tuple[int, int]],
        frame_width: int,
        frame_height: int
) -> CompositionScore:
    """analyze composition and provide score based on photography rules."""
    scores = {}
    recommendations = []

    # Rule of Thirds
    third_x = [frame_width // 3, 2 * frame_width // 3]
    third_y = [frame_height // 3, 2 * frame_height // 3]
    rot_intersections = [
        (third_x[0], third_y[0]), (third_x[1], third_y[0]),
        (third_x[0], third_y[1]), (third_x[1], third_y[1])
    ]

    if subject_positions:
        min_distances = []
        for subject_pos in subject_positions:
            distances = [
                np.sqrt((subject_pos[0] - ix) ** 2 + (subject_pos[1] - iy) ** 2)
                for ix, iy in rot_intersections
            ]
            min_distances.append(min(distances))

        frame_diagonal = np.sqrt(frame_width ** 2 + frame_height ** 2)
        normalized_distances = [d / frame_diagonal for d in min_distances]
        rot_score = max(0.0, min(1.0, 1.0 - min(normalized_distances)))

        if rot_score < 0.6:
            recommendations.append("Move subject closer to rule-of-thirds intersection point")
    else:
        rot_score = 0.5

    scores['rule_of_thirds'] = rot_score

    # Balance
    if len(subject_positions) > 0:
        center_x = np.mean([p[0] for p in subject_positions])
        center_y = np.mean([p[1] for p in subject_positions])

        deviation_x = abs(center_x - frame_width / 2) / frame_width
        deviation_y = abs(center_y - frame_height / 2) / frame_height

        optimal_deviation = 0.15
        balance_x = 1.0 - abs(deviation_x - optimal_deviation) / 0.5
        balance_y = 1.0 - abs(deviation_y - optimal_deviation) / 0.5
        balance_score = max(0.0, min(1.0, (balance_x + balance_y) / 2))

        if balance_score < 0.5:
            recommendations.append("Improve horizontal/vertical balance")
    else:
        balance_score = 0.5

    scores['balance'] = balance_score

    # Subject Placement
    edge_threshold = 0.1
    if subject_positions:
        edge_violations = sum(
            1 for x, y in subject_positions
            if (x < frame_width * edge_threshold or x > frame_width * (1 - edge_threshold) or
                y < frame_height * edge_threshold or y > frame_height * (1 - edge_threshold))
        )
        placement_score = 1.0 - (edge_violations / len(subject_positions))
        if placement_score < 0.8:
            recommendations.append("Move subjects away from frame edges")
    else:
        placement_score = 0.5

    scores['subject_placement'] = placement_score

    # Negative Space
    if subject_positions and len(frame.shape) == 3:
        subject_area_ratio = len(subject_positions) * 0.1
        if subject_area_ratio < 0.3:
            negative_space_score = 1.0
        elif subject_area_ratio < 0.5:
            negative_space_score = 0.8
        elif subject_area_ratio < 0.7:
            negative_space_score = 0.6
        else:
            negative_space_score = 0.4
            recommendations.append("Consider wider framing for better negative space")
    else:
        negative_space_score = 0.7

    scores['negative_space'] = negative_space_score

    overall_score = (
            scores['rule_of_thirds'] * 0.3 +
            scores['balance'] * 0.25 +
            scores['subject_placement'] * 0.25 +
            scores['negative_space'] * 0.2
    )

    return CompositionScore(
        overall_score=overall_score,
        rule_of_thirds_score=scores['rule_of_thirds'],
        balance_score=scores['balance'],
        subject_placement_score=scores['subject_placement'],
        negative_space_score=scores['negative_space'],
        recommendations=recommendations
    )


def apply_photography_rules(
        subject_positions: List[Dict[str, Any]],
        frame_dimensions: Tuple[int, int]
) -> Dict[str, Any]:
    """apply professional photography rules and provide recommendations."""
    width, height = frame_dimensions
    recommendations = {
        'optimal_framing': None,
        'adjustments_needed': [],
        'composition_rating': 0.0
    }

    if not subject_positions:
        return recommendations

    positions = [(s['center_x'], s['center_y']) for s in subject_positions]
    frame_mock = np.zeros((height, width, 3), dtype=np.uint8)
    comp_score = calculate_composition_score(frame_mock, positions, width, height)

    recommendations['composition_rating'] = comp_score.overall_score
    recommendations['adjustments_needed'] = comp_score.recommendations

    if len(subject_positions) == 1:
        recommendations['optimal_framing'] = "portrait"
    elif len(subject_positions) == 2:
        recommendations['optimal_framing'] = "couple"
    elif len(subject_positions) <= 5:
        recommendations['optimal_framing'] = "small_group"
    else:
        recommendations['optimal_framing'] = "large_group"

    return recommendations


def format_exposure_for_display(settings: ExposureSettings) -> str:
    """format exposure settings for human-readable display."""
    return (
        f"Exposure Settings:\n"
        f"  Aperture: f/{settings.aperture}\n"
        f"  Shutter: {settings.shutter_speed}\n"
        f"  ISO: {settings.iso}\n"
        f"  Lighting: {settings.lighting.value}\n"
        f"  Scene: {settings.scene_type.value}\n"
        f"  Confidence: {settings.confidence:.1%}\n"
        f"  Reasoning: {settings.reasoning}"
    )


def validate_exposure_settings(settings: ExposureSettings) -> Tuple[bool, List[str]]:
    """validate exposure settings are within camera capabilities."""
    warnings = []
    is_valid = True

    if settings.aperture < APERTURE_MIN or settings.aperture > APERTURE_MAX:
        warnings.append(f"Aperture f/{settings.aperture} out of range [{APERTURE_MIN}-{APERTURE_MAX}]")
        is_valid = False

    if settings.iso < ISO_MIN or settings.iso > ISO_MAX:
        warnings.append(f"ISO {settings.iso} out of range [{ISO_MIN}-{ISO_MAX}]")
        is_valid = False

    if settings.iso > ISO_ACCEPTABLE_MAX:
        warnings.append(f"ISO {settings.iso} exceeds quality threshold ({ISO_ACCEPTABLE_MAX})")

    return is_valid, warnings


def should_use_manual_exposure(config: Optional[PhotoUtilsConfig] = None) -> bool:
    """check if manual exposure control should be used based on config."""
    if config is None:
        config = _config
    return config.exposure_enabled


def reload_config() -> PhotoUtilsConfig:
    """reload configuration from settings.py."""
    global _config

    reload_exposure_config()
    _config = PhotoUtilsConfig()

    # Update module-level constants
    globals()['SENSOR_WIDTH_MM'] = _config.sensor_width_mm
    globals()['SENSOR_HEIGHT_MM'] = _config.sensor_height_mm
    globals()['CIRCLE_OF_CONFUSION_MM'] = _config.circle_of_confusion_mm
    globals()['APERTURE_MIN'] = _config.aperture_min
    globals()['APERTURE_MAX'] = _config.aperture_max
    globals()['ISO_MIN'] = _config.iso_min
    globals()['ISO_MAX'] = _config.iso_max
    globals()['ISO_ACCEPTABLE_MAX'] = _config.iso_acceptable_max

    logger.info("Config reloaded from settings.py")
    return _config


def get_current_config() -> PhotoUtilsConfig:
    """get current global configuration."""
    return _config
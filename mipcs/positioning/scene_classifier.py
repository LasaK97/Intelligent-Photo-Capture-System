import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import sys
from pathlib import Path

from mediapipe.tasks.python.benchmark.benchmark_utils import average
from scipy.integrate import lebedev_rule

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from .position_calculator import  PersonPosition
from ..vision.face_analyzer import FaceAnalysis
from config.settings import get_settings
from ..utils.logger import get_logger, log_performance
from ..utils.exceptions import SceneClassificationError, SceneAnalysisError

logger = get_logger(__name__)

class SceneType(Enum):
    """scene classification types"""
    UNKNOWN = auto()
    PORTRAIT = auto()
    COUPLE = auto()
    SMALL_GROUP = auto()
    MEDIUM_GROUP = auto()
    LARGE_GROUP = auto()
    SEPARATED_PEOPLE = auto()

class CompositionQuality(Enum):
    """Composition quality assessment"""
    EXCELLENT = auto()
    GOOD = auto()
    FAIR = auto()
    POOR = auto()
    UNACCEPTABLE = auto()

@dataclass
class SceneAnalysis:
    """Complete scene analysis and composition assessment"""

    #scene classification
    scene_type: SceneType
    person_count: int
    confidence: float                  # classification confidence (0-1)

    #spatial analysis
    group_centroid: Tuple[float, float, float]  #group center position
    horizontal_spread: float                    # width of group in meters
    depth_spread: float                         # Front to back spread in meters
    average_distance: float                     # average distance from camera

    #composition metrics
    composition_quality: CompositionQuality
    framing_score: float                       # how well framed the subject/s is (0-1)
    spacing_score: float                       # how well spaced the subject/s are (0-1)
    distance_score: float                      # how optimal the distance is (0-1)

    #guidance recommendations
    optimal_distance: float                  # recommended camera distance
    recommended_action_keys: List[str]        # action keys for voice guidance
    camera_adjustments: Dict[str, float]    # camera positioning adjustments

    #face analysis integration
    faces_detected: int
    faces_visible: int                      # faces facing the camera
    face_quality_average: float              # average face quality score (0-1

    #technical details
    analysis_timestamp: float
    processing_time_ms: float
    positions_used: List[PersonPosition]

@dataclass
class CompositionRules:
    """composition rules for different scene types"""

    #distance performances
    max_distance: float
    min_distance: float
    optimal_distance: float

    #spacing requirements
    min_person_separation: float        # min distance between people
    max_person_separation: float        # max distance before "separated"

    #frame composition
    horizontal_padding: float            # horizontal frame padding ratio
    vertical_padding: float              # vertical frame padding ratio

    #quality thresholds
    min_framing_score: float              # min acceptable framing
    min_spacing_score: float              # min acceptable spacing

class SceneClassifier:

    def __init__(self):
        """Initialize the scene classifier class"""
        self.settings = get_settings()

        #load composition rules
        self.composition_rules = self._load_composition_rules()

        #performance tracking
        self.analysis_stats = {
                    'scenes_analyzed': 0,
                    'total_analysis_time': 0.0,
                    'classification_accuracy': 0.0,  # ground truth for real accuracy
                    'average_processing_time_ms': 0.0
                }

        logger.info("scene_classifier_initialized", rules_loaded=len(self.composition_rules))

    def _load_composition_rules(self) -> Dict[SceneType, CompositionRules]:
        """load composition rules"""
        distance_config = self.settings.photo_capture.scene_classification.distance_ranges
        padding_config = self.settings.photo_capture.scene_classification.composition_padding['padding']
        rules = {
            SceneType.PORTRAIT: CompositionRules(
                min_distance=distance_config.portrait.min_distance,
                max_distance=distance_config.portrait.max_distance,
                optimal_distance=distance_config.portrait.optimal_distance,
                min_person_separation=0.0, # N/A for single person
                max_person_separation=0.0,
                horizontal_padding=padding_config.portrait.horizontal,
                vertical_padding=padding_config.portrait.vertical,
                min_framing_score=0.7,
                min_spacing_score=1.0,     # N/A for single person
            ),
            SceneType.COUPLE: CompositionRules(
                min_distance=distance_config.couple.min_distance,
                max_distance=distance_config.couple.max_distance,
                optimal_distance=distance_config.couple.optimal_distance,
                min_person_separation=0.2,
                max_person_separation=distance_config.couple.max_horizontal_spread,
                horizontal_padding=padding_config.couple.horizontal,
                vertical_padding=padding_config.couple.vertical,
                min_framing_score=0.7,
                min_spacing_score=0.6,
            ),
            SceneType.SMALL_GROUP: CompositionRules(
                min_distance=distance_config.small_group.min_distance,
                max_distance=distance_config.small_group.max_distance,
                optimal_distance=distance_config.small_group.optimal_distance,
                min_person_separation=0.3,
                max_person_separation=distance_config.small_group.max_horizontal_spread,
                horizontal_padding=padding_config.small_group.horizontal,
                vertical_padding=padding_config.small_group.vertical,
                min_framing_score=0.6,
                min_spacing_score=0.5,
            ),
            SceneType.MEDIUM_GROUP: CompositionRules(
                min_distance=distance_config.medium_group.min_distance,
                max_distance=distance_config.medium_group.max_distance,
                optimal_distance=distance_config.medium_group.optimal_distance,
                min_person_separation=0.4,
                max_person_separation=distance_config.medium_group.max_horizontal_spread,
                horizontal_padding=padding_config.medium_group.horizontal,
                vertical_padding=padding_config.medium_group.vertical,
                min_framing_score=0.5,
                min_spacing_score=0.4,
            ),
            SceneType.LARGE_GROUP: CompositionRules(
                min_distance=distance_config.large_group.min_distance,
                max_distance=distance_config.large_group.max_distance,
                optimal_distance=distance_config.large_group.optimal_distance,
                min_person_separation=0.5,
                max_person_separation=distance_config.large_group.max_horizontal_spread,
                horizontal_padding=padding_config.large_group.horizontal,
                vertical_padding=padding_config.large_group.vertical,
                min_framing_score=0.4,
                min_spacing_score=0.3,
            )
        }

        return rules

    @log_performance("scene_analysis")
    def analyze_scene(
        self,
        person_positions: List[PersonPosition],
        face_analyses: Optional[List[Optional[FaceAnalysis]]] = None,
    ) -> SceneAnalysis:

        """perform scene analysis"""

        start_time = time.time()

        if not person_positions:
            return self._create_empty_analysis(start_time)

        try:
            # STEP 1: Classify scene type
            scene_type, classification_confidence = self._classify_scene_type(person_positions)

            # STEP 2: Calculate spatial metrics
            spatial_metrics = self._calculate_spatial_metrics(person_positions)

            # STEP 3: Evaluate composition quality
            composition_scores = self._evaluate_composition(scene_type, spatial_metrics)

            # STEP 4: Generate guidance recommendations
            recommendations = self._generate_recommendations(scene_type, spatial_metrics, composition_scores)

            # STEP 5: Integrate face analysis if available
            face_metrics = self._analyze_faces(face_analyses) if face_analyses else {}

            # STEP 6: Determine final composition quality
            overall_quality = self._determine_overall_quality(composition_scores)

            # create complete analysis
            analysis = SceneAnalysis(
                #classification
                scene_type=scene_type,
                person_count=len(person_positions),
                confidence=classification_confidence,

                #spatial analysis
                group_centroid=spatial_metrics['centroid'],
                horizontal_spread=spatial_metrics['horizontal_spread'],
                depth_spread=spatial_metrics['depth_spread'],
                average_distance=spatial_metrics['average_distance'],

                #composition metrics
                composition_quality=overall_quality,
                framing_score=composition_scores['framing_score'],
                spacing_score=composition_scores['spacing_score'],
                distance_score=composition_scores['distance_score'],

                # guidance
                optimal_distance=recommendations['optimal_distance'],
                recommended_action_keys=recommendations['action_keys'],
                camera_adjustments=recommendations['camera_adjustments'],

                #face analysis
                faces_detected=face_metrics.get('faces_detected', 0),
                faces_visible=face_metrics.get('faces_visible', 0),
                face_quality_average=face_metrics.get('average_quality', 0.0),

                # technical details
                analysis_timestamp=time.time(),
                processing_time_ms=(time.time() - start_time) * 1000,
                positions_used=person_positions
            )

            #update stats
            self._update_analysis_stats(analysis.processing_time_ms)

            logger.debug("scene_analysis_complete",
                    scene_type=scene_type.name,
                    person_count=len(person_positions),
                    quality=overall_quality.name,
                    processing_time_ms=analysis.processing_time_ms)

            return analysis

        except Exception as e:
            logger.error("scene_analysis_failed", error=str(e))
            raise SceneAnalysisError("Failed to analyze scene") from e

    def _classify_scene_type(
            self,
            positions: List[PersonPosition]
    ) -> Tuple[SceneType, float]:
        """classify scene type based on person positions"""

        person_count = len(positions)

        if person_count == 1:
            return SceneType.PORTRAIT, 1.0

        elif person_count == 2:

            #check if they are close enough to be a couple
            pos1, pos2 = positions[0].position_3d, positions[1].position_3d
            horizontal_distance = abs(pos1[0] - pos2[0])    # X distance in camera frame

            couple_threshold = self.composition_rules[SceneType.COUPLE].max_person_separation

            if horizontal_distance <= couple_threshold:
                confidence = 1.0 - (horizontal_distance / couple_threshold) * 0.3 # high confidence if close
                return SceneType.COUPLE, max(0.7, confidence)
            else:
                return SceneType.SEPARATED_PEOPLE, 0.8

        elif person_count <= 4:
            # small group --> check whether they are grouped
            horizontal_spread = self._calculate_horizontal_spread(positions)
            group_threshold = self.composition_rules[SceneType.SMALL_GROUP].max_person_separation

            if horizontal_spread <= group_threshold:
                return SceneType.SMALL_GROUP, 0.9
            else:
                return SceneType.SEPARATED_PEOPLE, 0.7

        elif person_count <= 7:
            return SceneType.MEDIUM_GROUP, 0.8

        else:
            return SceneType.LARGE_GROUP, 0.8

    def _calculate_spatial_metrics(
            self,
            positions: List[PersonPosition]
    ) -> Dict[str, float]:
        """calculate spatial metrics """

        # extract 3D positions
        pos_3d = [pos.position_3d for pos in positions]
        distances = [pos.distance_from_camera for pos in positions]

        #centroid calculation
        centroid = (
            np.mean([p[0] for p in pos_3d]), # X --> horizontal
            np.mean([p[1] for p in pos_3d]), # Y --> vertical
            np.mean([p[2] for p in pos_3d])  # Z --> depth - in to the scene
        )

        # horizontal spread X -axis
        x_positions = [p[0] for p in pos_3d]
        horizontal_spread = max(x_positions) - min(x_positions) if len(x_positions) > 1 else 0.0

        # depth spread Z -axis
        z_positions = [p[2] for p in pos_3d]
        depth_spread = max(z_positions) - min(z_positions) if len(z_positions) > 1 else 0.0

        # average distance
        average_distance = np.mean(distances)

        return {
            'centroid': centroid,
            'horizontal_spread': horizontal_spread,
            'depth_spread': depth_spread,
            'average_distance': average_distance,
            'min_distance': min(distances),
            'max_distance': max(distances)
        }

    def _calculate_horizontal_spread(
            self,
            positions: List[PersonPosition]
    ) -> float:
        """calculate horizontal spread (X axis)"""
        if len(positions) <=1:
            return 0.0

        x_positions = [pos.position_3d[0] for pos in positions]
        return max(x_positions) - min(x_positions)

    def _evaluate_composition(
            self,
            scene_type: SceneType,
            spatial_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """evaluate composition quality"""

        if scene_type not in self.composition_rules:
            #unknown
            return {
                'framing_score': 0.5,
                'spacing_score': 0.5,
                'distance_score': 0.5
            }

        rules = self.composition_rules[scene_type]

        #distance score -> check how close to the optimal distance
        avg_distance = spatial_metrics['average_distance']
        optimal_distance = rules.optimal_distance

        if avg_distance < rules.min_distance:
            # too close
            distance_score = max(0.0, 0.5 * (avg_distance / rules.min_distance))
        elif avg_distance > rules.max_distance:
            # too far
            overshoot = avg_distance - rules.max_distance
            max_overshoot= rules.max_distance * 0.5     # allow 50% overshoot before score hits 0
            distance_score = max(0.0, 0.5 * (1 - (overshoot / max_overshoot)))
        else:
            # within acceptable range
            distance_error = abs(avg_distance - optimal_distance)
            acceptable_range = rules.max_distance - rules.min_distance
            distance_score = 1.0 - (distance_error / acceptable_range) * 0.5

        # spacing score
        if scene_type == SceneType.PORTRAIT:
            spacing_score = 1.0 # N/A for single person
        else:
            horizontal_spread = spatial_metrics['horizontal_spread']

            if horizontal_spread <= rules.min_person_separation:
                # too crowded
                spacing_score = horizontal_spread / rules.min_person_separation

            elif horizontal_spread > rules.max_person_separation:
                # too spread out
                excess = horizontal_spread - rules.max_person_separation
                max_excess = rules.max_person_separation   # allow up to 2x before score hits 0
                spacing_score = max(0.0, 1.0 - (excess / max_excess))

            else:
                # good spacing
                spacing_score = 1.0


        # framing score --> how well subjects fit in expected frame
        # TODO: for now estimate --> ideally need actual frame dimentions --> Assume 70 deg HFOV --> FIX WITH TESTING

        horizontal_spread = spatial_metrics['horizontal_spread']
        avg_distance = spatial_metrics['average_distance']

        #estimate frame with at this distance
        estimated_frame_width = 2 * avg_distance * np.tan(np.radians(35))

        # calculate padding
        if estimated_frame_width > 0:
            used_frame_ratio = horizontal_spread / estimated_frame_width
            ideal_ratio = 1.0 - 2 * rules.horizontal_padding      # padding both sides

            if used_frame_ratio < ideal_ratio * 0.5:
                # subjects too small in frame
                framing_score = used_frame_ratio / (ideal_ratio * 0.5)
            elif used_frame_ratio > 1.0:
                # subjects too large for frame
                framing_score = max(0.0, 1.0 - (used_frame_ratio - 1.0))
            else:
                # good framing
                framing_error = abs(used_frame_ratio - ideal_ratio)
                framing_score = 1.0 - framing_error
        else:
            framing_score = 0.5   # Fallback

        return {
            'framing_score': max(0.0, min(1.0, framing_score)),
            'spacing_score': max(0.0, min(1.0, spacing_score)),
            'distance_score': max(0.0, min(1.0, distance_score)),
        }

    def _generate_recommendations(
            self,
            scene_type: SceneType,
            spatial_metrics: Dict[str, float],
            composition_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """generate action recommendation to improve the compositions"""

        movements = []
        camera_adjustments = {}

        if scene_type not in self.composition_rules:
            return {
                'optimal_distance': 4.0,    #default
                'action_keys': [],
                'camera_adjustments': {}
            }

        rules = self.composition_rules[scene_type]
        avg_distance = spatial_metrics['average_distance']
        horizontal_spread = spatial_metrics['horizontal_spread']

        #distance recommendations
        if avg_distance < rules.min_distance:
            movements.append('move_further')
            camera_adjustments['move_back'] = rules.optimal_distance - avg_distance
        elif avg_distance > rules.max_distance:
            movements.append('move_closer')
            camera_adjustments['move_forward'] = avg_distance - rules.optimal_distance

        #spacing recommendations
        if scene_type != SceneType.PORTRAIT:
            if horizontal_spread < rules.min_person_separation:
                movements.append('spread_out')
            elif horizontal_spread > rules.max_person_separation:
                movements.append('move_together')

        # framing recommendations based on centroid
        centroid_x = spatial_metrics['centroid'][0]
        if abs(centroid_x) > 0.5:   #50 cm off center
            if centroid_x > 0:
                movements.append('move_left')
                camera_adjustments['pan_left'] = np.degrees(np.arctan(centroid_x / avg_distance))
            else:
                movements.append('move_right')
                camera_adjustments['pan_right'] = np.degrees(np.arctan(abs(centroid_x) / avg_distance))

        #quality based recommendation
        if composition_scores['distance_score'] > 0.8 and composition_scores['spacing_score'] > 0.8:
            movements = ["perfect_position"]
        elif not movements:
            # movements =  ["Please adjust your positions slightly"]
            pass

        return {
            'optimal_distance': rules.optimal_distance,
            'action_keys': movements,
            'camera_adjustments': camera_adjustments
        }

    def _analyze_faces(
            self,
            face_analyses: List[Optional[FaceAnalysis]]
    ) -> Dict[str, Any]:
        """analyze face detection results"""
        valid_analyses = [f for f in face_analyses if f is not None]

        faces_detected = len(valid_analyses)

        faces_visible = sum(1 for f in valid_analyses if f.facing_camera)

        if valid_analyses:
            average_quality = np.mean([f.confidence for f in valid_analyses])
        else:
            average_quality = 0.0

        return {
            'faces_detected': faces_detected,
            'faces_visible': faces_visible,
            'average_quality': average_quality,
        }

    def _determine_overall_quality(
            self,
            composition_scores: Dict[str, float],
    )-> CompositionQuality:
        """determine the overall quality of the scene"""

        #weighted score

        # Priority --> 1. distance score , 2. framing score , 3.spacing score
        overall_score = (
            composition_scores['distance_score'] * 0.4 +   #
            composition_scores['framing_score'] * 0.35 +
            composition_scores['spacing_score'] * 0.25
        )

        if overall_score >= 0.9:
            return CompositionQuality.EXCELLENT
        elif overall_score >= 0.7:
            return CompositionQuality.GOOD
        elif overall_score >= 0.5:
            return CompositionQuality.FAIR
        elif overall_score >= 0.3:
            return CompositionQuality.POOR
        else:
            return CompositionQuality.UNACCEPTABLE


    def _create_empty_analysis(self, start_time: float) -> SceneAnalysis:
        """Create analysis for empty scene."""

        return SceneAnalysis(
            scene_type=SceneType.UNKNOWN,
            person_count=0,
            confidence=1.0,
            group_centroid=(0.0, 0.0, 0.0),
            horizontal_spread=0.0,
            depth_spread=0.0,
            average_distance=0.0,
            composition_quality=CompositionQuality.UNACCEPTABLE,
            framing_score=0.0,
            spacing_score=0.0,
            distance_score=0.0,
            optimal_distance=4.0,
            recommended_action_keys=["welcome"],
            camera_adjustments={},
            faces_detected=0,
            faces_visible=0,
            face_quality_average=0.0,
            analysis_timestamp=time.time(),
            processing_time_ms=(time.time() - start_time) * 1000,
            positions_used=[]
        )

    def _update_analysis_stats(self, processing_time_ms: float) -> None:
        """update analysis statistics."""
        self.analysis_stats['scenes_analyzed'] += 1
        self.analysis_stats['total_analysis_time'] += processing_time_ms

        # Calculate average processing time
        count = self.analysis_stats['scenes_analyzed']
        avg_time = self.analysis_stats['total_analysis_time'] / count
        self.analysis_stats['average_processing_time_ms'] = avg_time

    def get_analysis_stats(self) -> Dict[str, Any]:
        """get current analysis statistics"""
        return {
            **self.analysis_stats,
            'composition_rules_count': len(self.composition_rules)
        }


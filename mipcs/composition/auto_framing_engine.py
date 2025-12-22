import asyncio
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from pathlib import Path
import sys

from triton.experimental.gluon.language.amd.cdna4.async_copy import async_wait

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from ..utils.logger import get_logger, log_performance

from .framing_calculator import (
    FramingCalculator,
    FramingStrategy as FCStrategy,
    FramingResult as FCResult,
    SubjectGroup as FCSubjectGroup
)
from .background_analyzer import (
    BackgroundAnalyzer,
    BackgroundAnalysis,
    BackgroundQuality
)
from .shot_optimizer import (
    ShotOptimizer,
    OptimizationStrategy,
    ShotCandidate,
    ShotSequence
)
from ..control.motion_planner import (
    MotionPlanner,
    MotionProfile,
    Trajectory,
    Waypoint
)
from .composition_analyzer import CompositionAnalyzer

from ..utils.geometry_utils import Point3D

logger = get_logger(__name__)


class FramingMode(Enum):
    """Auto-framing operational modes"""
    SINGLE_SHOT = "single_shot"
    MULTI_SHOT = "multi_shot"
    CONTINUOUS = "continuous"
    CREATIVE_MODE = "creative_mode"


class FramingPriority(Enum):
    """Framing optimization priorities"""
    COMPOSITION_QUALITY = "composition_quality"
    EXECUTION_SPEED = "execution_speed"
    SUBJECT_VISIBILITY = "subject_visibility"
    BACKGROUND_HARMONY = "background_harmony"
    CREATIVE_VARIETY = "creative_variety"


class FramingStatus(Enum):
    """Current framing engine status"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FramingRequest:
    """Complete framing request"""
    subject_groups: List[Dict]
    scene_analysis: Dict
    mode: FramingMode = FramingMode.SINGLE_SHOT
    priority: FramingPriority = FramingPriority.COMPOSITION_QUALITY
    max_execution_time: float = 20.0
    max_shots: int = 3
    quality_threshold: float = 0.7
    metadata: Dict = field(default_factory=dict)


@dataclass
class FramingResult:
    """Complete framing result"""
    primary_shot: Optional[Dict] = None
    shot_sequence: Optional[List[Dict]] = None
    trajectory: Optional[Dict] = None
    overall_quality: float = 0.0
    composition_score: float = 0.0
    execution_time: float = 0.0
    status: str = "completed"
    background_analysis: Optional[Dict] = None
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class FramingPlan:
    """Complete framing plan for optimal shot composition"""
    target_position: Tuple[float, float, float]
    focal_distance: float
    aperture_setting: float
    composition_score: float
    strategy: str
    confidence: float
    execution_time_s: float
    fallback_plans: List = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class SubjectGroup:
    """Group of subjects to be framed together"""
    subjects: List[Dict]
    center_position: Tuple[float, float, float]
    bounding_box_3d: Tuple[float, float, float, float, float, float]
    group_type: str
    importance_score: float
    stability: float

class AutoFramingEngine:

    def __init__(self):

        settings = get_settings()
        self.core_config = settings.auto_framing.core
        self.composition_config = settings.auto_framing.composition
        self.exposure_config = settings.auto_framing.exposure
        self.camera_config = settings.camera
        self.gimbal_config = settings.gimbal

        self._init_components()

        # engine status
        self.current_status = FramingStatus.IDLE
        self.current_request: Optional[FramingRequest] = None
        self.last_result: Optional[FramingResult] = None

        # position tracking for motion prediction
        self.position_history = {}   # track_id --> deque of (timestamp, position)
        self.velocity_estimates = {} # track_id --> (vx, vy, vz)
        self.max_history_length = 10

        # performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.total_processing_time = 0.0
        self.component_performance = {
            'background_analysis': {'count': 0, 'total_time': 0.0},
            'composition_analysis': {'count': 0, 'total_time': 0.0},
            'shot_optimization': {'count': 0, 'total_time': 0.0},
            'motion_planning': {'count': 0, 'total_time': 0.0},
            'framing_calculation': {'count': 0, 'total_time': 0.0}
        }

        # caching
        self.analysis_cache = {}
        self.cache_max_size = self.core_config.cache_size

        # quality thresholds
        self.quality_thresholds = {
            'minimum_composition': self.composition_config.quality_gates.minimum_acceptable,
            'target_composition': self.composition_config.quality_gates.target_quality,
            'excellent_composition': self.composition_config.quality_gates.excellent_quality
        }

        logger.info(
            "auto_framing_engine_initialized",
            enabled=self.core_config.enabled,
            default_mode=self.core_config.default_mode,
            quality_target=self.composition_config.quality_gates.target_quality
        )

    def _init_components(self):
        """initialize components"""
        self.framing_calculator = FramingCalculator()
        self.background_analyzer = BackgroundAnalyzer()
        self.composition_analyzer = CompositionAnalyzer()
        self.shot_optimizer = ShotOptimizer()
        self.motion_planner = MotionPlanner()


    # position tracking
    def update_position_tracking(
            self,
            track_id: int,
            position: Tuple[float, float, float],
            timestamp: float
    ) -> None:
        """update position history for tracking and prediction"""

        if track_id not in self.position_history:
            self.position_history[track_id] = deque(maxlen=self.max_history_length)

        self.position_history[track_id].append((timestamp, position))

        # update velocity estimate --> if only have enough history
        if len(self.position_history[track_id]) >= 2:
            self._update_velocity_estimate(track_id)

    def _update_velocity_estimate(self, track_id: int) -> None:
        """calculate velocity from position history"""
        history = self.position_history[track_id]

        if len(history) < 2:
            return

        # use last 2 positions
        (t1, pos1) = history[-2]
        (t2, pos2) = history[-1]

        dt = t2 - t1
        if dt > 0:
            vx = (pos2[0] - pos1[0]) / dt
            vy = (pos2[1] - pos1[1]) / dt
            vz = (pos2[2] - pos1[2]) / dt
            self.velocity_estimates[track_id] = (vx, vy, vz)

    def predict_position(
            self,
            track_id: int,
            dt: float
    ) -> Optional[Tuple[float, float, float]]:
        """predict position based on velocity estimates"""

        if track_id not in self.position_history or track_id not in self.velocity_estimates:
            return None

        #get last known position
        last_timestamp, last_position = self.position_history[track_id][-1]
        vx, vy, vz = self.velocity_estimates[track_id]

        # linear prediction --> Simple
        predicted = (
            last_position[0] + vx * dt,
            last_position[1] + vy * dt,
            last_position[2] + vz * dt
        )

        return predicted



    async def process_framing_request(
            self,
            request: FramingRequest
    ) -> FramingResult:
        """process complete framing request"""

        start_time = time.perf_counter()
        self.current_status = FramingStatus.ANALYZING
        self.current_request = request
        self.total_requests += 1

        try:
            # update position tracking
            for group in request.subject_groups:
                for subject in group.get('subjects', []):
                    track_id = subject.get('track_id')
                    if track_id and 'position_3d' in subject:
                        self.update_position_tracking(
                            track_id,
                            subject['position_3d'],
                            start_time
                        )

            # 1. Analyze the background
            bg_start = time.perf_counter()
            background_analysis = await self._analyze_background(
                request.scene_analysis.get('frame'),
                request.subject_groups
            )
            self._track_component_time('background_analysis', time.perf_counter() - bg_start)

            # 2. calculate optimal framing
            frame_start = time.perf_counter()
            framing_result = await self._calculate_framing(
                request.subject_groups[0] if request.subject_groups else None,
                request.scene_analysis,
                request.priority
            )
            self._track_component_time('framing_calculation', time.perf_counter() - frame_start)

            # 3. Optimize shots if multi-shots mode
            shot_sequence = None
            if request.mode == FramingMode.MULTI_SHOT:
                opt_start = time.perf_counter()
                shot_sequence = await self._optimize_shots(
                    framing_result,
                    request.scene_analysis,
                    request
                )
                self._track_component_time('shot_optimization', time.perf_counter() - opt_start)


            # 4. Plan motion trajectory
            motion_start = time.perf_counter()
            trajectory = await self._plan_motion(framing_result, request)
            self._track_component_time('motion_planning', time.perf_counter() - motion_start)

            # build result
            result = FramingResult(
                primary_shot={
                    'pan': framing_result['pan'],
                    'tilt': framing_result['tilt'],
                    'zoom': framing_result['zoom'],
                    'composition_score': framing_result.get('score', 0.7),
                    'strategy': framing_result.get('strategy', 'unknown')
                },
                shot_sequence=shot_sequence,
                trajectory=trajectory,
                overall_quality=framing_result.get('score', 0.7),
                composition_score=framing_result.get('score', 0.7),
                execution_time=time.perf_counter() - start_time,
                status='completed',
                background_analysis=background_analysis,
                recommendations=self._generate_recommendations(
                    framing_result,
                    background_analysis
                ),
                metadata={
                    'background_quality': background_analysis.get('quality_rating'),
                    'strategy_used': framing_result.get('strategy', 'unknown')
                }
            )

            self.successful_requests += 1
            self.last_result = result
            self.current_status = FramingStatus.COMPLETED

            logger.info(
                "framing_request_completed",
                quality=result.overall_quality,
                strategy=result.primary_shot.get('strategy'),
                time_ms=round(result.execution_time * 1000, 1)
            )

            return result

        except Exception as e:
            logger.error("framing_request_failed", error=str(e), exc_info=True)
            self.current_status = FramingStatus.FAILED

            # error result
            return FramingResult(
                status='failed',
                execution_time=time.perf_counter() - start_time,
                metadata={'error': str(e)}
            )

    async def _analyze_background(
            self,
            frame: np.ndarray,
            subject_groups: List[Dict]
    ) -> Dict:
        """analyze background frame --> use BackgroundAnalyzer"""
        if frame is None:
            return {'quality_rating': 'unknown', 'overall_score': 0.5}

        try:
            bg_analysis = await self.background_analyzer.analyze_background(
                frame = frame,
                subject_mask= None,
                region_of_interest= None,
            )
            return {
                'quality_rating': bg_analysis.quality_rating.value,
                'overall_score': bg_analysis.overall_score,
                'edge_density': bg_analysis.edge_density,
                'color_variance': bg_analysis.color_variance,
                'distraction_score': bg_analysis.distraction_score,
                'recommendations': bg_analysis.recommendations
            }
        except Exception as e:
            logger.warning(f"background_analysis_failed: {e}")
            return {'quality_rating': 'unknown', 'overall_score': 0.5}

    async def _calculate_framing(
            self,
            primary_group: Optional[Dict],
            scene_analysis: Dict,
            priority: FramingPriority
    ) -> Dict:
        """calculate optimal framing --> use FramingCalculator"""
        if not primary_group:
            return {'pan': 0.0, 'tilt': 0.0, 'zoom': 50.0, 'score': 0.3, 'strategy': 'default'}

        try:
            # convert to FCSubjectGroup
            positions = self._extract_positions(primary_group)
            if not positions:
                return {'pan': 0.0, 'tilt': 0.0, 'zoom': 50.0, 'score': 0.3, 'strategy': 'default'}

            fc_group = FCSubjectGroup(
                positions_3d=[Point3D(*p) for p in positions],
                center=Point3D(*primary_group.get('center_position', (0, 0, 5))),
                spread=self._calculate_spread(primary_group),
                scene_type=primary_group.get('group_type', 'unknown')
            )

            # select strategy based on priority
            strategy = self._select_strategy_from_priority(priority, fc_group)

            # calculate framing
            fc_result = await self.framing_calculator.calculate_optimal_framing(
                subject_group=fc_group,
                current_pose=None,
                strategy=strategy,
                constraints=None
            )

            return {
                'pan': fc_result.pan_angle,
                'tilt': fc_result.tilt_angle,
                'zoom': fc_result.zoom_level,
                'score': fc_result.composition_score,
                'strategy': fc_result.strategy_used.value,
                'confidence': fc_result.confidence
            }

        except Exception as e:
            logger.warning(f"framing_request_failed: {e}")
            return {'pan': 0.0, 'tilt': 0.0, 'zoom': 50.0, 'score': 0.5, 'strategy': 'fallback'}

    async def _optimize_shots(
            self,
            framing_result: Dict,
            scene_analysis: Dict,
            request: FramingRequest
    ) -> List[Dict]:
        """optimize shot sequence --> use ShotOptimizer"""
        try:
            base_framing = {
                'pan_angle': framing_result['pan'],
                'tilt_angle': framing_result['tilt'],
                'zoom_level': framing_result['zoom']
            }

            sequence = await self.shot_optimizer.optimize_shot_sequence(
                base_framing=base_framing,
                scene_analysis=scene_analysis,
                strategy=OptimizationStrategy.BALANCED,
                constraints={'max_time': request.max_execution_time}
            )

            return [
                {
                    'pan': shot.pan_angle,
                    'tilt': shot.tilt_angle,
                    'zoom': shot.zoom_level,
                    'shot_type': shot.shot_type.value,
                    'quality': shot.composition_score
                }
                for shot in sequence.shots
            ]

        except Exception as e:
            logger.warning(f"shot_optimization_failed: {e}")
            return []

    async def _plan_motion(
            self,
            framing_result: Dict,
            request: FramingRequest
    ) -> Dict:
        """plan motion trajectory --> use MotionPlanner"""

        try:
            current_pose = {'pan': 0.0, 'tilt': 0.0, 'roll': 0.0, 'zoom': 50.0}
            target_pose = {
                'pan': framing_result['pan'],
                'tilt': framing_result['tilt'],
                'roll': 0.0,
                'zoom': framing_result['zoom']
            }

            trajectory = await self.motion_planner.plan_trajectory(
                start_pose=current_pose,
                target_pose=target_pose,
                duration=2.0,
                profile=MotionProfile.EASE_IN_OUT
            )

            return {
                'waypoints': [
                    {
                        'pan': wp.pan,
                        'tilt': wp.tilt,
                        'timestamp': wp.timestamp
                    }
                    for wp in trajectory.waypoints
                ],
                'duration': trajectory.total_duration,
                'smoothness': trajectory.smoothness_score,
                'safe': trajectory.safe
            }

        except Exception as e:
            logger.warning(f"motion_planning_failed: {e}")
            return {'waypoints': [], 'duration': 0.0, 'smoothness': 0.0, 'safe': False}

    def extract_positions(
            self,
            group: Dict,
    ) -> List[Tuple[float, float, float]]:
        """extract 3D positions from subject group"""
        positions = []
        for subject in group.get('subjects', []):
            if 'position_3d' in subject:
                positions.append(subject['position_3d'])
        return positions

    def _calculate_spread(
            self,
            group: Dict,
    ) -> float:
        """calculate spread of subject group"""
        positions = self._extract_positions(group)
        if len(positions) < 2:
            return 0.0

        positions_array = np.array(positions)
        center = np.mean(positions_array, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in positions_array]
        return max(distances)

    def _select_strategy_from_priority(
            self,
            priority: FramingPriority,
            group: FCSubjectGroup
    ) -> FCStrategy:
        """select strategy based on priority"""
        if priority == FramingPriority.COMPOSITION_QUALITY:
            return FCStrategy.AUTO
        elif priority == FramingPriority.EXECUTION_SPEED:
            return FCStrategy.CENTER_WEIGHTED
        elif priority == FramingPriority.CREATIVE_VARIETY:
            return FCStrategy.GOLDEN_RATIO
        else:
            return FCStrategy.AUTO

    def _generate_recommendations(
            self,
            framing_result: Dict,
            background_analysis: Dict
    ) -> List[str]:
        """generate recommendations based on analysis."""
        recommendations = []

        # quality-based recommendations
        if framing_result.get('score', 1.0) < self.quality_thresholds['minimum_composition']:
            recommendations.append("Composition quality below minimum - consider repositioning")

        # background recommendations
        if background_analysis.get('distraction_score', 0) > 0.7:
            recommendations.append("Background has distracting elements - change angle or position")

        if background_analysis.get('edge_density', 0) > 0.5:
            recommendations.append("Background is busy - consider simpler backdrop")

        # add background-specific recommendations
        bg_recs = background_analysis.get('recommendations', [])
        recommendations.extend(bg_recs[:2])  # Add top 2

        return recommendations[:5]  # limit = 5

    def _track_component_time(
            self,
            component: str,
            elapsed: float
    ) -> None:
        """Track component performance"""
        if component in self.component_performance:
            self.component_performance[component]['count'] += 1
            self.component_performance[component]['total_time'] += elapsed

    async def analyze_and_plan_framing(
            self,
            frame: np.ndarray,
            person_detections: List[Dict],
            positions_3d: Dict[int, Tuple[float, float, float]],
            depth_image: Optional[np.ndarray] = None,
            scene_type: Optional[str] = None
    ) -> FramingPlan:
        """ analyze scene and generate framing plan"""

        # subject groups
        subject_groups = self._group_subjects_from_detections(
            person_detections,
            positions_3d
        )

        # create request
        request = FramingRequest(
            subject_groups=subject_groups,
            scene_analysis={'frame': frame, 'depth': depth_image, 'scene_type': scene_type}
        )

        # process
        result = await self.process_framing_request(request)

        # convert to FramingPlan
        if result.primary_shot:
            return FramingPlan(
                target_position=(
                    result.primary_shot['pan'],
                    result.primary_shot['tilt'],
                    result.primary_shot['zoom']
                ),
                focal_distance=5.0,
                aperture_setting=2.8,
                composition_score=result.composition_score,
                strategy=result.primary_shot.get('strategy', 'unknown'),
                confidence=result.primary_shot.get('confidence', 0.7),
                execution_time_s=result.execution_time,
                metadata=result.metadata
            )
        else:
            # default plan
            return FramingPlan(
                target_position=(0.0, 0.0, 50.0),
                focal_distance=5.0,
                aperture_setting=5.6,
                composition_score=0.3,
                strategy='default',
                confidence=0.5,
                execution_time_s=1.0
            )

    def _group_subjects_from_detections(
            self,
            person_detections: List[Dict],
            positions_3d: Dict[int, Tuple[float, float, float]],
    ) -> List[Dict]:
        """group subjects based on spatial relationships"""
        if not person_detections:
            return []

        groups = []
        unassigned = list(person_detections)

        while unassigned:
            current = unassigned.pop(0)
            group_subjects = [current]

            person_id = current.get('person_id', 0)
            if person_id in positions_3d:
                continue

            group_center = list(positions_3d[person_id])

            # find nearby subjects
            remaining = []

            for subject in unassigned:
                subject_id = subject.get('person_id', 0)
                if subject_id not in positions_3d:
                    remaining.append(subject)
                    continue

                subject_pos = positions_3d[subject_id]
                distance = np.linalg.norm(
                    np.array(subject_pos) - np.array(group_center)
                )

                threshold = self.core_config.grouping_distance_base

                if distance < threshold:
                    group_subjects.append(subject)

                    #update center
                    positions = [positions_3d[s.get('person_id', 0)] for s in group_subjects if s.get('person_id', 0) in positions_3d]
                    if positions:
                        group_center = list(np.mean(positions, axis=0))
                else:
                    remaining.append(subject)

            unassigned = remaining

            # create group dist
            positions = [positions_3d[s.get('person_id', 0)] for s in group_subjects if s.get('person_id', 0) in positions_3d]
            if positions:
                positions_array = np.array(positions)
                center_position = tuple(np.mean(positions_array, axis=0))

                num_subjects = len(group_subjects)
                if num_subjects == 1:
                    group_type = "individual"
                elif num_subjects == 2:
                    group_type = "couple"
                elif num_subjects <= 5:
                    group_type = "small_group"
                else:
                    group_type = "large_group"

                groups.append({
                    'subjects': group_subjects,
                    'center_position': center_position,
                    'group_type': group_type
                })

        return groups

    def get_performance_stats(self):
        """get performance statistics"""
        if self.total_requests == 0:
            return {'total_requests': 0}

        stats = {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'success_rate': self.successful_requests / self.total_requests,
            'avg_processing_time_ms': (self.total_processing_time / self.total_requests) * 1000,
            'components': {}
        }

        for component, data in self.component_performance.items():
            if data['count'] > 0:
                stats['components'][component] = {
                    'count': data['count'],
                    'avg_time_ms': (data['total_time'] / data['count']) * 1000
                }

        return stats

# convenience function
async def analyze_and_frame_scene(
        frame: np.ndarray,
        person_detections: List[Dict],
        positions_3d: Dict[int, Tuple[float, float, float]],
        **kwargs
) -> FramingPlan:
    """convenience function for auto-framing analysis."""
    engine = AutoFramingEngine()
    return await engine.analyze_and_plan_framing(
        frame,
        person_detections,
        positions_3d,
        **kwargs
    )


import numpy as np
import asyncio
import time
from typing import Dict, Tuple, List, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ..utils.logger import get_logger
from ..utils.geometry_utils import (
    Point2D,
    Point3D,
    calculate_rule_of_thirds_points,
    calculate_golden_ratio_points,
    calculate_centroid_3d,
    calculate_weighted_centroid_3d,
    calculate_2d_distance
)

from config.settings import get_settings

logger = get_logger(__name__)

class FramingStrategy(Enum):
    """Photography framing strategies"""
    RULE_OF_THIRDS = "rule_of_thirds"
    GOLDEN_RATIO = "golden_ratio"
    CENTER_WEIGHTED = "center_weighted"
    FILL_FRAME = "fill_frame"
    DYNAMIC_SYMMETRY = "dynamic_symmetry"
    AUTO = "auto"

@dataclass
class SubjectGroup:
    """Group of subjects to frame"""
    positions_3d: List[Point3D]
    center: Point3D
    spread: float
    weights: Optional[List[float]] = None
    scene_type: str = "unknown"

@dataclass
class FramingResult:
    """Camera framing result"""
    pan_angle: float  # degrees
    tilt_angle: float  # degrees
    zoom_level: float  # mm focal length
    composition_score: float  # 0-1
    rule_alignment: float  # 0-1
    subject_coverage: float  # 0-1
    strategy_used: FramingStrategy
    confidence: float  # 0-1
    reasoning: str
    alternatives: List[Dict] = field(default_factory=list)

class FramingCalculator:

    def __init__(self):
        settings = get_settings()

        # configs
        self.composition = settings.auto_framing.composition
        self.weights = self.composition.weights
        self.quality_gates = self.composition.quality_gates
        self.auto_framing_weights = self.composition.auto_framing_weights

        # cam specs
        self.camera_config = settings.camera
        self.camera_specs = settings.hardware_specs.camera

        # sensor and lens specs
        self.sensor_width = self.camera_specs.sensor.width_mm
        self.sensor_height = self.camera_specs.sensor.height_mm
        self.min_focal_length = self.camera_specs.lens.focal_length.min
        self.max_focal_length = self.camera_specs.lens.focal_length.max

        # frame dimensions
        self.frame_width = 1920
        self.frame_height = 1080

        # photography constant
        self.phi = 1.618033988749895  # Golden ratio

        # performance tracking
        self.calculation_times = []
        self.strategy_usage = {s: 0 for s in FramingStrategy}

        logger.info(
            "framing_calculator_initialized",
            sensor_mm=f"{self.sensor_width}x{self.sensor_height}",
            focal_range_mm=f"{self.min_focal_length}-{self.max_focal_length}",
            quality_target=self.quality_gates.target_quality
        )

    async def calculate_optimal_framing(
            self,
            subject_group: SubjectGroup,
            current_pose: Optional[Dict[str, float]] = None,
            strategy: FramingStrategy = FramingStrategy.AUTO,
            constraints: Optional[Dict] = None
    ) -> FramingResult:
        """calculate optimal camera framing."""
        start_time = time.time()

        # auto-select strategy
        if strategy == FramingStrategy.AUTO:
            strategy = self._select_strategy(subject_group)
            logger.debug(f"auto_strategy_selected={strategy.value}")

        # calculate based on strategy
        if strategy == FramingStrategy.RULE_OF_THIRDS:
            result = await self._rule_of_thirds(subject_group)
        elif strategy == FramingStrategy.GOLDEN_RATIO:
            result = await self._golden_ratio(subject_group)
        elif strategy == FramingStrategy.CENTER_WEIGHTED:
            result = await self._center_weighted(subject_group)
        elif strategy == FramingStrategy.FILL_FRAME:
            result = await self._fill_frame(subject_group)
        elif strategy == FramingStrategy.DYNAMIC_SYMMETRY:
            result = await self._dynamic_symmetry(subject_group)
        else:
            result = await self._center_weighted(subject_group)

        # apply constraints
        if constraints:
            result = self._apply_constraints(result, constraints)

        # optimize from current position
        if current_pose:
            result = self._optimize_movement(result, current_pose)

        # track performance
        elapsed = time.time() - start_time
        self.calculation_times.append(elapsed)
        self.strategy_usage[strategy] += 1

        logger.debug(
            "framing_calculated",
            strategy=strategy.value,
            pan=f"{result.pan_angle:.1f}°",
            tilt=f"{result.tilt_angle:.1f}°",
            zoom=f"{result.zoom_level:.0f}mm",
            score=f"{result.composition_score:.2f}",
            time_ms=f"{elapsed * 1000:.1f}"
        )

        return result

    def _select_strategy(self, group: SubjectGroup) -> FramingStrategy:
        """auto-select best strategy for scene"""
        num_subjects = len(group.positions_3d)
        spread = group.spread

        # large groups (wide spread) → fill frame
        if num_subjects >= 5 or spread > 3.0:
            return FramingStrategy.FILL_FRAME

        # couples → dynamic symmetry
        if num_subjects == 2:
            return FramingStrategy.DYNAMIC_SYMMETRY

        # small groups → fill frame
        if num_subjects >= 3:
            return FramingStrategy.FILL_FRAME

        # single subject → rule of thirds (professional standard) --> TODO : RESEARCH --> important
        return FramingStrategy.RULE_OF_THIRDS

    async def _rule_of_thirds(self, group: SubjectGroup) -> FramingResult:
        """rule of thirds -> place subject at 1/3 or 2/3 intersection"""

        # intersection points (normalized [0-1] )
        intersections = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]

        # calculate centroid
        if group.weights:
            centroid = calculate_weighted_centroid_3d(group.positions_3d, group.weights)
        else:
            centroid = calculate_centroid_3d(group.positions_3d)

        # choose intersection: lower-left for subjects on left, lower-right for right
        target = intersections[2] if centroid.x < 0 else intersections[3]

        # calculate angles
        pan, tilt = self._angles_to_point(centroid, target)

        # Calculate zoom
        zoom = self._zoom_for_subjects(group, padding=0.15)

        # Evaluate
        score = self._evaluate_composition(group, target, FramingStrategy.RULE_OF_THIRDS)

        return FramingResult(
            pan_angle=pan,
            tilt_angle=tilt,
            zoom_level=zoom,
            composition_score=score,
            rule_alignment=0.95,
            subject_coverage=0.70,
            strategy_used=FramingStrategy.RULE_OF_THIRDS,
            confidence=0.90,
            reasoning="Professional rule of thirds composition"
        )

    async def _golden_ratio(self, group: SubjectGroup) -> FramingResult:
        """golden ratio --> phi = 1.618 intersections"""

        # golden points
        golden_x = [1 / self.phi, 1 - 1 / self.phi]  # [0.382, 0.618]
        golden_y = [1 / self.phi, 1 - 1 / self.phi]
        intersections = [(x, y) for x in golden_x for y in golden_y]

        centroid = calculate_centroid_3d(group.positions_3d)

        # prefer 0.618 position (more natural)
        target = intersections[3] if centroid.x >= 0 else intersections[1]

        pan, tilt = self._angles_to_point(centroid, target)
        zoom = self._zoom_for_subjects(group, padding=0.12)
        score = self._evaluate_composition(group, target, FramingStrategy.GOLDEN_RATIO)

        return FramingResult(
            pan_angle=pan,
            tilt_angle=tilt,
            zoom_level=zoom,
            composition_score=score,
            rule_alignment=0.92,
            subject_coverage=0.75,
            strategy_used=FramingStrategy.GOLDEN_RATIO,
            confidence=0.88,
            reasoning="Aesthetically pleasing golden ratio composition"
        )

    async def _center_weighted(self, group: SubjectGroup) -> FramingResult:
        """center-weighted -> slightly off-center for balance"""

        centroid = calculate_centroid_3d(group.positions_3d)
        target = (0.5, 0.55)  # slightly lower for headroom

        pan, tilt = self._angles_to_point(centroid, target)
        zoom = self._zoom_for_subjects(group, padding=0.18)
        score = self._evaluate_composition(group, target, FramingStrategy.CENTER_WEIGHTED)

        return FramingResult(
            pan_angle=pan,
            tilt_angle=tilt,
            zoom_level=zoom,
            composition_score=score,
            rule_alignment=0.75,
            subject_coverage=0.65,
            strategy_used=FramingStrategy.CENTER_WEIGHTED,
            confidence=0.80,
            reasoning="Safe center-weighted composition"
        )

    async def _fill_frame(self, group: SubjectGroup) -> FramingResult:
        """fill frame: maximize frame usage for groups"""

        # calculate bounding box
        x_coords = [p.x for p in group.positions_3d]
        y_coords = [p.y for p in group.positions_3d]
        z_coords = [p.z for p in group.positions_3d]

        center = Point3D(
            (min(x_coords) + max(x_coords)) / 2,
            (min(y_coords) + max(y_coords)) / 2,
            (min(z_coords) + max(z_coords)) / 2
        )

        target = (0.5, 0.5)  # center for groups

        pan, tilt = self._angles_to_point(center, target)
        zoom = self._zoom_for_subjects(group, padding=0.08)  # Tight framing
        score = self._evaluate_composition(group, target, FramingStrategy.FILL_FRAME)

        return FramingResult(
            pan_angle=pan,
            tilt_angle=tilt,
            zoom_level=zoom,
            composition_score=score,
            rule_alignment=0.70,
            subject_coverage=0.85,
            strategy_used=FramingStrategy.FILL_FRAME,
            confidence=0.85,
            reasoning=f"Fill-frame for {len(group.positions_3d)} subjects"
        )

    async def _dynamic_symmetry(self, group: SubjectGroup) -> FramingResult:
        """dynamic symmetry -> balanced placement for couples"""

        if len(group.positions_3d) != 2:
            return await self._center_weighted(group)

        # midpoint of two subjects
        s1, s2 = group.positions_3d
        midpoint = Point3D(
            (s1.x + s2.x) / 2,
            (s1.y + s2.y) / 2,
            (s1.z + s2.z) / 2
        )

        target = (0.5, 0.5)

        pan, tilt = self._angles_to_point(midpoint, target)
        zoom = self._zoom_for_subjects(group, padding=0.15)
        score = self._evaluate_composition(group, target, FramingStrategy.DYNAMIC_SYMMETRY)

        return FramingResult(
            pan_angle=pan,
            tilt_angle=tilt,
            zoom_level=zoom,
            composition_score=score,
            rule_alignment=0.88,
            subject_coverage=0.75,
            strategy_used=FramingStrategy.DYNAMIC_SYMMETRY,
            confidence=0.92,
            reasoning="Balanced symmetry for couple"
        )

    def _angles_to_point(
            self,
            subject_pos: Point3D,
            target_frame: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Calculate pan/tilt to place subject at target frame position"""

        x, y, z = subject_pos.x, subject_pos.y, subject_pos.z

        # standard pan calculation
        pan_standard = np.degrees(np.arctan2(y, x))

        # manriix inverted yaw axis
        pan = -pan_standard

        # tilt calculation
        horizontal_dist = np.sqrt(x ** 2 + y ** 2)
        tilt = np.degrees(np.arctan2(z, horizontal_dist))

        # adjust for target position offset from center
        target_x, target_y = target_frame

        # approximate FOV  TODO: Use actual focal length in production
        h_fov = 62.2  # degrees
        v_fov = 39.6  # degrees

        # offset from center (0.5, 0.5)
        offset_x = (target_x - 0.5) * h_fov
        offset_y = (0.5 - target_y) * v_fov  # Inverted Y

        pan += offset_x
        tilt += offset_y

        return pan, tilt

    def _zoom_for_subjects(
            self,
            group: SubjectGroup,
            padding: float = 0.15
    ) -> float:
        """calculate zoom to frame subjects with padding"""

        if len(group.positions_3d) == 1:
            # single subject - assume 1.7m height, fill ~70% of frame
            distance = group.positions_3d[0].z
            required_fov = 1.7 / (distance * (1 - padding))
        else:
            # multiple subjects - use spread
            spread = group.spread
            avg_distance = np.mean([p.z for p in group.positions_3d])
            required_fov = spread / (avg_distance * (1 - padding))

        # calculate focal length from FOV
        # focal_length = sensor_width / (2 * tan(FOV/2))
        focal_length = self.sensor_width / (2 * np.tan(required_fov / 2))

        # clamp to lens limits
        focal_length = np.clip(focal_length, self.min_focal_length, self.max_focal_length)

        return float(focal_length)

    def _evaluate_composition(
            self,
            group: SubjectGroup,
            target: Tuple[float, float],
            strategy: FramingStrategy
    ) -> float:
        """evaluate composition quality (0-1)"""

        # Base scores per strategy
        base_scores = {
            FramingStrategy.RULE_OF_THIRDS: 0.85,
            FramingStrategy.GOLDEN_RATIO: 0.90,
            FramingStrategy.CENTER_WEIGHTED: 0.75,
            FramingStrategy.FILL_FRAME: 0.80,
            FramingStrategy.DYNAMIC_SYMMETRY: 0.88
        }
        score = base_scores.get(strategy, 0.75)

        # bonus for ideal strategy
        num_subjects = len(group.positions_3d)
        if num_subjects == 1 and strategy == FramingStrategy.RULE_OF_THIRDS:
            score += 0.05
        elif num_subjects == 2 and strategy == FramingStrategy.DYNAMIC_SYMMETRY:
            score += 0.05
        elif num_subjects >= 3 and strategy == FramingStrategy.FILL_FRAME:
            score += 0.03

        return min(1.0, max(0.0, score))

    def _apply_constraints(self, result: FramingResult, constraints: Dict) -> FramingResult:
        """apply movement constraints"""
        if 'max_pan' in constraints:
            result.pan_angle = np.clip(result.pan_angle, -constraints['max_pan'], constraints['max_pan'])
        if 'max_tilt' in constraints:
            result.tilt_angle = np.clip(result.tilt_angle, -constraints['max_tilt'], constraints['max_tilt'])
        if 'max_zoom' in constraints:
            result.zoom_level = min(result.zoom_level, constraints['max_zoom'])
        if 'min_zoom' in constraints:
            result.zoom_level = max(result.zoom_level, constraints['min_zoom'])
        return result

    def _optimize_movement(self, result: FramingResult, current: Dict[str, float]) -> FramingResult:
        """optimize to minimize movement from current position"""
        current_pan = current.get('pan', 0.0)
        current_tilt = current.get('tilt', 0.0)

        pan_delta = abs(result.pan_angle - current_pan)
        tilt_delta = abs(result.tilt_angle - current_tilt)

        # Small adjustments
        if pan_delta < 2.0 and tilt_delta < 2.0:
            result.reasoning += " (minimal adjustment)"
            result.confidence *= 0.95

        return result

    def get_performance_stats(self) -> Dict:
        """get performance statistics"""
        if not self.calculation_times:
            return {}

        return {
            'avg_time_ms': np.mean(self.calculation_times) * 1000,
            'max_time_ms': np.max(self.calculation_times) * 1000,
            'total_calculations': len(self.calculation_times),
            'strategy_usage': {k.value: v for k, v in self.strategy_usage.items()}
        }



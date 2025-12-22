import numpy as np
import asyncio
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ShotType(Enum):
    """Types of camera shots"""
    WIDE = "wide"  # Establishes context
    MEDIUM = "medium"  # Standard shot
    CLOSE_UP = "close_up"  # Portrait/detail
    DUTCH_ANGLE = "dutch_angle"  # Creative tilt   # TODO: RESEARCH on camera shots
    LOW_ANGLE = "low_angle"  # Dramatic from below
    HIGH_ANGLE = "high_angle"  # Overview from above
    OVER_SHOULDER = "over_shoulder"  # For conversations

class OptimizationStrategy(Enum):
    """Shot optimization strategies"""
    QUALITY_FIRST = "quality_first"  # Best composition scores
    VARIETY_FIRST = "variety_first"  # Diverse shot types
    SPEED_FIRST = "speed_first"  # Minimal camera movement
    BALANCED = "balanced"  # Mix of all factors

@dataclass
class ShotCandidate:
    """Candidate camera shot"""
    shot_type: ShotType
    pan_angle: float
    tilt_angle: float
    zoom_level: float
    roll_angle: float = 0.0  # For dutch angles

    # Quality metrics
    composition_score: float = 0.0
    feasibility_score: float = 0.0
    variety_score: float = 0.0
    technical_score: float = 0.0
    aesthetic_score: float = 0.0

    # Metadata
    estimated_time: float = 0.0  # Seconds to achieve
    movement_distance: float = 0.0  # Total movement required
    reasoning: str = ""

@dataclass
class ShotSequence:
    """Optimized sequence of shots"""
    shots: List[ShotCandidate]
    total_time: float
    total_quality: float
    variety_score: float
    efficiency_score: float
    strategy_used: OptimizationStrategy
    recommendations: List[str] = field(default_factory=list)

class ShotOptimizer:

    def __init__(self):
        settings = get_settings()

        # config
        self.core_config = settings.auto_framing.core
        self.quality_threshold = self.core_config.quality_threshold
        self.max_execution_time = self.core_config.max_execution_time

        # composition config for scoring
        self.composition = settings.auto_framing.composition

        # shot type parameters
        self.shot_parameters = {
            ShotType.WIDE: {'zoom_factor': 0.7, 'tilt_offset': 0},
            ShotType.MEDIUM: {'zoom_factor': 1.0, 'tilt_offset': 0},
            ShotType.CLOSE_UP: {'zoom_factor': 1.5, 'tilt_offset': 0},
            ShotType.DUTCH_ANGLE: {'zoom_factor': 1.0, 'roll': 15},
            ShotType.LOW_ANGLE: {'zoom_factor': 1.0, 'tilt_offset': -15},
            ShotType.HIGH_ANGLE: {'zoom_factor': 1.0, 'tilt_offset': 15},
            ShotType.OVER_SHOULDER: {'zoom_factor': 1.2, 'pan_offset': 30}
        }

        # performance tracking
        self.optimization_times = []
        self.shots_generated = 0

        logger.info(
            "shot_optimizer_initialized",
            quality_threshold=self.quality_threshold,
            max_time=self.max_execution_time
        )

    async def optimize_shot_sequence(
            self,
            base_framing: Dict,
            scene_analysis: Dict,
            strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
            constraints: Optional[Dict] = None
    ) -> ShotSequence:
        """Optimize sequence of shots"""
        start_time = time.time()

        # generate candidate shots
        candidates = await self._generate_candidates(base_framing, scene_analysis)

        # score all candidates
        scored_candidates = await self._score_candidates(candidates, scene_analysis)

        # filter by quality threshold
        quality_candidates = [
            c for c in scored_candidates
            if c.composition_score >= self.quality_threshold
        ]

        if not quality_candidates:
            # lower threshold if no candidates meet criteria
            logger.warning("no_quality_shots_lowering_threshold")
            quality_candidates = scored_candidates[:3]  # Take top 3

        # optimize sequence based on strategy
        if strategy == OptimizationStrategy.QUALITY_FIRST:
            optimized = self._optimize_for_quality(quality_candidates)
        elif strategy == OptimizationStrategy.VARIETY_FIRST:
            optimized = self._optimize_for_variety(quality_candidates)
        elif strategy == OptimizationStrategy.SPEED_FIRST:
            optimized = self._optimize_for_speed(quality_candidates)
        else:  # BALANCED
            optimized = self._optimize_balanced(quality_candidates)

        # apply constraints
        if constraints:
            optimized = self._apply_constraints(optimized, constraints)

        # calculate sequence metrics
        sequence = self._build_sequence(optimized, strategy)

        # track performance
        elapsed = time.time() - start_time
        self.optimization_times.append(elapsed)
        self.shots_generated += len(candidates)

        logger.debug(
            "shot_sequence_optimized",
            strategy=strategy.value,
            num_shots=len(sequence.shots),
            total_quality=f"{sequence.total_quality:.2f}",
            variety=f"{sequence.variety_score:.2f}",
            time_ms=f"{elapsed * 1000:.1f}"
        )

        return sequence

    async def _generate_candidates(
            self,
            base_framing: Dict,
            scene_analysis: Dict
    ) -> List[ShotCandidate]:
        """generate candidate shots from different angles and types"""

        candidates = []
        base_pan = base_framing.get('pan_angle', 0.0)
        base_tilt = base_framing.get('tilt_angle', 0.0)
        base_zoom = base_framing.get('zoom_level', 50.0)

        num_subjects = len(scene_analysis.get('subject_groups', []))

        # 1. Wide shot (context)
        candidates.append(ShotCandidate(
            shot_type=ShotType.WIDE,
            pan_angle=base_pan,
            tilt_angle=base_tilt + self.shot_parameters[ShotType.WIDE]['tilt_offset'],
            zoom_level=base_zoom * self.shot_parameters[ShotType.WIDE]['zoom_factor'],
            reasoning="Wide shot for context"
        ))

        # 2. Medium shot (standard)
        candidates.append(ShotCandidate(
            shot_type=ShotType.MEDIUM,
            pan_angle=base_pan,
            tilt_angle=base_tilt,
            zoom_level=base_zoom,
            reasoning="Medium shot as primary"
        ))

        # 3. Close-up (portrait)
        if num_subjects <= 2:
            candidates.append(ShotCandidate(
                shot_type=ShotType.CLOSE_UP,
                pan_angle=base_pan,
                tilt_angle=base_tilt,
                zoom_level=base_zoom * self.shot_parameters[ShotType.CLOSE_UP]['zoom_factor'],
                reasoning="Close-up for portrait"
            ))

        # 4. Low angle (dramatic)
        candidates.append(ShotCandidate(
            shot_type=ShotType.LOW_ANGLE,
            pan_angle=base_pan,
            tilt_angle=base_tilt + self.shot_parameters[ShotType.LOW_ANGLE]['tilt_offset'],
            zoom_level=base_zoom,
            reasoning="Low angle for drama"
        ))

        # 5. High angle (overview)
        candidates.append(ShotCandidate(
            shot_type=ShotType.HIGH_ANGLE,
            pan_angle=base_pan,
            tilt_angle=base_tilt + self.shot_parameters[ShotType.HIGH_ANGLE]['tilt_offset'],
            zoom_level=base_zoom,
            reasoning="High angle for overview"
        ))

        # 6. Dutch angle (creative)
        candidates.append(ShotCandidate(
            shot_type=ShotType.DUTCH_ANGLE,
            pan_angle=base_pan,
            tilt_angle=base_tilt,
            zoom_level=base_zoom,
            roll_angle=self.shot_parameters[ShotType.DUTCH_ANGLE]['roll'],
            reasoning="Dutch angle for creativity"
        ))

        # 7. Over-shoulder (for conversations)
        if num_subjects == 2:
            # Left shoulder
            candidates.append(ShotCandidate(
                shot_type=ShotType.OVER_SHOULDER,
                pan_angle=base_pan + self.shot_parameters[ShotType.OVER_SHOULDER]['pan_offset'],
                tilt_angle=base_tilt,
                zoom_level=base_zoom * self.shot_parameters[ShotType.OVER_SHOULDER]['zoom_factor'],
                reasoning="Over-shoulder left"
            ))

            # Right shoulder
            candidates.append(ShotCandidate(
                shot_type=ShotType.OVER_SHOULDER,
                pan_angle=base_pan - self.shot_parameters[ShotType.OVER_SHOULDER]['pan_offset'],
                tilt_angle=base_tilt,
                zoom_level=base_zoom * self.shot_parameters[ShotType.OVER_SHOULDER]['zoom_factor'],
                reasoning="Over-shoulder right"
            ))

        return candidates

    async def _score_candidates(
            self,
            candidates: List[ShotCandidate],
            scene_analysis: Dict
    ) -> List[ShotCandidate]:
        """score all candidates across multiple criteria"""

        for candidate in candidates:
            # 1. composition score (based on shot type and scene)
            candidate.composition_score = self._score_composition(candidate, scene_analysis)

            # 2. feasibility score (can we actually achieve this shot?)
            candidate.feasibility_score = self._score_feasibility(candidate)

            # 3. variety score (how different from others?)
            candidate.variety_score = 0.8  # Updated later in context of all shots

            # 4. technical score (focus, exposure, stability)
            candidate.technical_score = self._score_technical(candidate)

            # 5. aesthetic score (background, lighting)
            candidate.aesthetic_score = self._score_aesthetic(candidate, scene_analysis)

            # estimate time to achieve shot
            candidate.estimated_time = self._estimate_time(candidate)

            # calculate movement distance
            candidate.movement_distance = self._calculate_movement(candidate)

        return

    def _score_composition(self, shot: ShotCandidate, scene: Dict) -> float:
        """score composition quality"""

        # Base scores per shot type
        base_scores = {
            ShotType.WIDE: 0.75,
            ShotType.MEDIUM: 0.85,
            ShotType.CLOSE_UP: 0.88,
            ShotType.DUTCH_ANGLE: 0.70,
            ShotType.LOW_ANGLE: 0.78,
            ShotType.HIGH_ANGLE: 0.76,
            ShotType.OVER_SHOULDER: 0.82
        }

        score = base_scores.get(shot.shot_type, 0.75)

        # Adjust based on scene
        num_subjects = len(scene.get('subject_groups', []))

        if num_subjects == 1:
            if shot.shot_type == ShotType.CLOSE_UP:
                score += 0.08  # Bonus for portraits
        elif num_subjects == 2:
            if shot.shot_type == ShotType.OVER_SHOULDER:
                score += 0.10  # Bonus for conversations
        elif num_subjects >= 3:
            if shot.shot_type == ShotType.WIDE:
                score += 0.07  # Bonus for groups

        return min(1.0, score)

    def _score_feasibility(self, shot: ShotCandidate) -> float:
        """score shot feasibility (movement limits, stability)"""

        # Check angle limits
        if abs(shot.pan_angle) > 180:
            return 0.3  # out of range
        if abs(shot.tilt_angle) > 45:
            return 0.5  # difficult angle
        if abs(shot.roll_angle) > 20:
            return 0.6  # unstable roll

        # high tilts are less stable
        stability_penalty = abs(shot.tilt_angle) / 90 * 0.2

        return max(0.0, 1.0 - stability_penalty)

    def _score_technical(self, shot: ShotCandidate) -> float:
        """score technical quality (focus, exposure)"""

        # Close-ups need good focus
        if shot.shot_type == ShotType.CLOSE_UP:
            return 0.85

        # Wide shots easier technically
        if shot.shot_type == ShotType.WIDE:
            return 0.90

        # Dutch angles can have focus issues
        if shot.shot_type == ShotType.DUTCH_ANGLE:
            return 0.75

        return 0.82

    def _score_aesthetic(self, shot: ShotCandidate, scene: Dict) -> float:
        """score aesthetic quality (background, lighting)"""

        # Use background analysis if available
        bg_analysis = scene.get('background_analysis', {})
        bg_score = bg_analysis.get('overall_score', 0.75)

        # some shot types emphasize background more
        if shot.shot_type == ShotType.WIDE:
            return bg_score * 1.0  # background very important
        elif shot.shot_type == ShotType.CLOSE_UP:
            return 0.7 + bg_score * 0.3  # background less important
        else:
            return 0.65 + bg_score * 0.35

    def _estimate_time(self, shot: ShotCandidate) -> float:
        """estimate time to achieve shot (seconds)"""

        # base time per shot type
        base_times = {
            ShotType.WIDE: 1.5,
            ShotType.MEDIUM: 1.0,
            ShotType.CLOSE_UP: 2.0,  # needs focus adjustment
            ShotType.DUTCH_ANGLE: 2.5,  # needs roll adjustment
            ShotType.LOW_ANGLE: 2.0,
            ShotType.HIGH_ANGLE: 2.0,
            ShotType.OVER_SHOULDER: 1.8
        }

        base_time = base_times.get(shot.shot_type, 1.5)

        # add movement time
        movement_time = shot.movement_distance / 30  # 30°/s typical speed

        return base_time + movement_time

    def _calculate_movement(self, shot: ShotCandidate) -> float:
        """calculate total movement distance (degrees)"""

        # combine pan, tilt, roll movements
        movement = abs(shot.pan_angle) + abs(shot.tilt_angle) + abs(shot.roll_angle)

        return movement

    def _optimize_for_quality(self, candidates: List[ShotCandidate]) -> List[ShotCandidate]:
        """optimize for highest composition scores"""

        # sort by composition score
        sorted_shots = sorted(
            candidates,
            key=lambda s: s.composition_score,
            reverse=True
        )

        # take top 3-5 shots
        return sorted_shots[:5]

    def _optimize_for_variety(self, candidates: List[ShotCandidate]) -> List[ShotCandidate]:
        """optimize for diverse shot types"""

        # ensure one of each major type
        selected = []
        used_types = set()

        # priority order for variety
        priority_types = [
            ShotType.MEDIUM,
            ShotType.WIDE,
            ShotType.CLOSE_UP,
            ShotType.LOW_ANGLE,
            ShotType.HIGH_ANGLE
        ]

        for shot_type in priority_types:
            # find best shot of this type
            type_shots = [s for s in candidates if s.shot_type == shot_type]
            if type_shots:
                best = max(type_shots, key=lambda s: s.composition_score)
                selected.append(best)
                used_types.add(shot_type)

        # add any remaining high-quality shots
        for shot in sorted(candidates, key=lambda s: s.composition_score, reverse=True):
            if shot.shot_type not in used_types and len(selected) < 5:
                selected.append(shot)
                used_types.add(shot.shot_type)

        return selected

    def _optimize_for_speed(self, candidates: List[ShotCandidate]) -> List[ShotCandidate]:
        """optimize for minimal camera movement"""

        # sort by movement distance
        sorted_shots = sorted(
            candidates,
            key=lambda s: s.movement_distance
        )

        # take shots with minimal movement that still meet quality
        selected = []
        for shot in sorted_shots:
            if shot.composition_score >= self.quality_threshold and len(selected) < 3:
                selected.append(shot)

        return selected if selected else sorted_shots[:3]

    def _optimize_balanced(self, candidates: List[ShotCandidate]) -> List[ShotCandidate]:
        """balance quality, variety, and speed"""

        # calculate balanced score
        for shot in candidates:
            shot.variety_score = self._calculate_variety(shot, candidates)

            balanced_score = (
                    shot.composition_score * 0.4 +
                    shot.variety_score * 0.3 +
                    shot.feasibility_score * 0.2 +
                    (1.0 - shot.movement_distance / 180) * 0.1
            )
            shot.balanced_score = balanced_score

        # sort by balanced score
        sorted_shots = sorted(
            candidates,
            key=lambda s: s.balanced_score,
            reverse=True
        )

        return sorted_shots[:4]

    def _calculate_variety(self, shot: ShotCandidate, all_candidates: List[ShotCandidate]) -> float:
        """calculate how unique this shot is"""

        # count shots of same type
        same_type_count = sum(1 for s in all_candidates if s.shot_type == shot.shot_type)

        # variety = inverse of frequency
        variety = 1.0 - (same_type_count - 1) / max(1, len(all_candidates))

        return variety

    def _apply_constraints(
            self,
            shots: List[ShotCandidate],
            constraints: Dict
    ) -> List[ShotCandidate]:
        """apply time/movement constraints"""

        max_time = constraints.get('max_time', self.max_execution_time)
        max_shots = constraints.get('max_shots', 5)

        # trim to max shots
        shots = shots[:max_shots]

        # check time constraint
        total_time = sum(s.estimated_time for s in shots)
        if total_time > max_time:
            # remove lowest quality shots until within time
            shots.sort(key=lambda s: s.composition_score, reverse=True)
            while sum(s.estimated_time for s in shots) > max_time and len(shots) > 1:
                shots.pop()

        return shots

    def _build_sequence(
            self,
            shots: List[ShotCandidate],
            strategy: OptimizationStrategy
    ) -> ShotSequence:
        """build final shot sequence with metrics"""

        # order shots for efficient movement
        ordered_shots = self._order_for_efficiency(shots)

        # calculate metrics
        total_time = sum(s.estimated_time for s in ordered_shots)
        total_quality = np.mean([s.composition_score for s in ordered_shots])

        # calculate variety (unique shot types)
        unique_types = len(set(s.shot_type for s in ordered_shots))
        variety_score = unique_types / len(ShotType)

        # calculate efficiency (quality per second)
        efficiency_score = total_quality / max(1, total_time)

        # generate recommendations
        recommendations = self._generate_recommendations(ordered_shots, strategy)

        return ShotSequence(
            shots=ordered_shots,
            total_time=total_time,
            total_quality=total_quality,
            variety_score=variety_score,
            efficiency_score=efficiency_score,
            strategy_used=strategy,
            recommendations=recommendations
        )

    def _order_for_efficiency(self, shots: List[ShotCandidate]) -> List[ShotCandidate]:
        """order shots to minimize total movement"""

        if len(shots) <= 1:
            return shots

        # simple greedy ordering: always move to nearest next shot
        ordered = [shots[0]]
        remaining = shots[1:]

        while remaining:
            current = ordered[-1]

            # find nearest shot
            nearest = min(
                remaining,
                key=lambda s: abs(s.pan_angle - current.pan_angle) + abs(s.tilt_angle - current.tilt_angle)
            )

            ordered.append(nearest)
            remaining.remove(nearest)

        return ordered

    def _generate_recommendations(
            self,
            shots: List[ShotCandidate],
            strategy: OptimizationStrategy
    ) -> List[str]:
        """generate recommendations for shot sequence"""

        recommendations = []

        if len(shots) == 1:
            recommendations.append("Single shot selected - consider adding variety")

        if all(s.shot_type in [ShotType.MEDIUM, ShotType.WIDE] for s in shots):
            recommendations.append("Consider adding creative angles (low, high, dutch)")

        total_time = sum(s.estimated_time for s in shots)
        if total_time > self.max_execution_time:
            recommendations.append(f"Sequence exceeds time limit ({total_time:.1f}s > {self.max_execution_time}s)")

        avg_quality = np.mean([s.composition_score for s in shots])
        if avg_quality < 0.7:
            recommendations.append("Consider improving shot composition or changing location")

        if not recommendations:
            recommendations.append("Shot sequence is well optimized")

        return recommendations

    def get_performance_stats(self) -> Dict:
        """get performance statistics"""
        if not self.optimization_times:
            return {}

        return {
            'avg_time_ms': np.mean(self.optimization_times) * 1000,
            'max_time_ms': np.max(self.optimization_times) * 1000,
            'total_optimizations': len(self.optimization_times),
            'shots_generated': self.shots_generated
        }
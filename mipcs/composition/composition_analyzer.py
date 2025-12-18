import numpy as np
import cv2
import asyncio
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ..utils.logger import get_logger
from ..utils.geometry_utils import (
    Point2D,
    calculate_intersection_points,
    calculate_rule_of_thirds_points,
    calculate_golden_ratio_points,
    find_nearest_composition_point,
    find_leading_lines,
    calculate_line_angle,
    detect_symmetry,
    calculate_2d_distance
)
from config.settings import get_settings, get_exposure_config


try:
    from ..positioning.transform_manager import TransformManager, Point3D
    TRANSFORM_MANAGER_AVAILABLE = True
except ImportError:
    TRANSFORM_MANAGER_AVAILABLE = False
    TransformManager = None
    Point3D = None

logger = get_logger(__name__)

class CompositionRule(Enum):
    """Available composition rules"""
    RULE_OF_THIRDS = "rule_of_thirds"
    GOLDEN_RATIO = "golden_ratio"
    DIAGONAL_LINES = "diagonal_lines"
    SYMMETRY = "symmetry"
    LEADING_LINES = "leading_lines"
    FRAMING = "framing"
    PATTERNS = "patterns"

@dataclass
class CompositionElement:
    """Detected compositional element in frame"""
    element_type: str  # "line", "shape", "area", "point"
    coordinates: List[Tuple[int, int]]  # Pixel coords
    strength: float  #strength of the element [0 - 1]
    composition_impact: float  # How much it affects composition
    rule_alignment: List[CompositionRule]  # Which rules it supports

@dataclass
class CompositionAnalysis:
    """Complete composition analysis result"""
    frame_dimensions: Tuple[int, int]  # (h, w)
    rule_scores: Dict[CompositionRule, float]  # Score for each rule
    detected_elements: List[CompositionElement]
    grid_intersections: List[Tuple[int, int]]  # Key composition points
    leading_lines: List[List[Tuple[int, int]]]  # Detected leading lines
    symmetry_axes: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    overall_score: float  # overall composition quality [0 - 1]
    recommendations: List[str]  # Improvement suggestions

class CompositionAnalyzer:
    """"""

    def __init__(
            self,
            transform_manager: Optional['TransformManager'] = None,
            camera_intrinsics: Optional[Dict] = None
    ):
        self.settings = get_exposure_config()
        self.config = self.settings.composition.analyzer
        self.transform_manager = transform_manager

        # camera intrinsics
        self.camera_intrinsics = camera_intrinsics or {
            'fx': 600.0,
            'fy': 600.0,
            'cx': 320.0,
            'cy': 240.0
        }
        # cache
        self.last_analysis: Optional[CompositionAnalysis] = None
        self.analysis_cache = {}

        logger.info(
            "composition_analyzer_initialized",
            use_transform_manager=transform_manager is not None,
            circle_detection_enabled=self.config.circle_detection.enabled
            )

    async def analyze_composition(
            self,
            frame: np.ndarray,
            subject_groups: List[Dict],
            existing_elements: Optional[List[CompositionElement]] = None,
    ) -> CompositionAnalysis:
        """Comprehensive analysis of composition"""

        height, width = frame.shape[:2]

        #detect compositional elements
        detected_elements = await self._detect_compositional_elements(frame)

        if existing_elements:
            detected_elements.extend(existing_elements)

        # calculate rule scores
        rule_scores = await self._calculate_rule_scores(
            frame, subject_groups, detected_elements
        )

        # find key intersection points
        grid_intersections = self._calculate_grid_intersections(height, width)

        # detect leading lines
        leading_lines =  await self._detect_leading_lines(frame)

        # find symmetry axes
        symmetry_axes = await self._detect_symmetry_axes(frame)

        # calculate overall score
        overall_score = self._calculate_overall_composition_score(rule_scores)

        # generate recommendations
        recommendations = await self._generate_composition_recommendations(
            rule_scores, subject_groups, detected_elements
        )

        analysis = CompositionAnalysis(
            frame_dimensions=(height, width),
            rule_scores=rule_scores,
            detected_elements=detected_elements,
            grid_intersections=grid_intersections,
            leading_lines=leading_lines,
            symmetry_axes=symmetry_axes,
            overall_score=overall_score,
            recommendations=recommendations
        )

        self.last_analysis = analysis

        logger.debug(
            "composition_analysis_completed",
            overall_score=overall_score,
            elements_detected=len(detected_elements),
            leading_lines=len(leading_lines),
            rule_scores={rule.value: round(score, 3) for rule, score in rule_scores.items()}
        )

        return analysis

    def _project_3d_to_2d(
            self,
            point_3d_tuple: Tuple[float, float, float],
            frame_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Project 3d point to 2d pixel coords"""

        height, width = frame_shape

        # sse transform manager (if available)
        if self.transform_manager and TRANSFORM_MANAGER_AVAILABLE:
            try:
                # convert to Point3D
                point_3d = Point3D(*point_3d_tuple)

                # transform to camera frame
                point_camera = self.transform_manager.transform_point(
                    point_3d,
                    source_frame="base_link",
                    target_frame="camera_color_optical_frame"
                )

                if point_camera is not None:
                    # project using camera intrinsics (pinhole model)
                    fx = self.camera_intrinsics['fx']
                    fy = self.camera_intrinsics['fy']
                    cx = self.camera_intrinsics['cx']
                    cy = self.camera_intrinsics['cy']

                    if point_camera.z > 0:  # point in front of camera
                        x_pixel = int((point_camera.x / point_camera.z) * fx + cx)
                        y_pixel = int((point_camera.y / point_camera.z) * fy + cy)

                        # clamp to frame boundaries
                        x_pixel = np.clip(x_pixel, 0, width - 1)
                        y_pixel = np.clip(y_pixel, 0, height - 1)

                        return x_pixel, y_pixel
            except Exception as e:
                logger.warning(
                    "transform_projection_failed_using_fallback",
                    error=str(e)
                )

        # fallback: Simplified projection -> backward compatible behavior
        center_x = int(width // 2 + point_3d_tuple[0] * 100)
        center_y = int(height // 2 - point_3d_tuple[1] * 100)

        # clamp to frame
        center_x = np.clip(center_x, 0, width - 1)
        center_y = np.clip(center_y, 0, height - 1)

        return center_x, center_y

    async def _detect_compositional_elements(self, frame: np.ndarray) -> List[CompositionElement]:
        """detect compositional elements in the frame"""

        elements = []

        # convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect edges using Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.line_detection_threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # calculate line properties
                line_coords = [(x1, y1), (x2, y2)]
                angle = calculate_line_angle(line_coords)
                length = calculate_2d_distance(Point2D(x1, y1), Point2D(x2, y2))

                # determine line strength based on length
                strength = min(1.0, length / (frame.shape[1] * 0.5))

                # determine which composition rules this line supports
                rule_alignment = self._analyze_line_composition_impact(
                    (x1, y1), (x2, y2), frame.shape[:2], angle
                )

                element = CompositionElement(
                    element_type="line",
                    coordinates=line_coords,
                    strength=strength,
                    composition_impact=len(rule_alignment) * 0.2,
                    rule_alignment=rule_alignment
                )

                elements.append(element)

        # detect circular/curved elements using config-based parameters
        if self.config.circle_detection.enabled:
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=self.config.circle_detection.min_radius * 2,
                param1=self.config.circle_detection.param1,
                param2=self.config.circle_detection.param2,
                minRadius=self.config.circle_detection.min_radius,
                maxRadius=self.config.circle_detection.max_radius
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Normalize radius to strength [0 - 1]
                    strength = min(1.0, r / self.config.circle_detection.max_radius)

                    element = CompositionElement(
                        element_type="shape",
                        coordinates=[(x, y)],  # center point
                        strength=strength,
                        composition_impact=0.3,
                        rule_alignment=[CompositionRule.PATTERNS, CompositionRule.FRAMING]
                    )
                    elements.append(element)

                logger.debug("circles_detected", count=len(circles))

        return elements

    def _analyze_line_composition_impact(
            self,
            start: Tuple[int, int],
            end: Tuple[int, int],
            frame_shape: Tuple[int, int],
            angle: float
    ) -> List[CompositionRule]:
        """analyze line impact for composition rules"""
        height, width = frame_shape
        rules = []

        # check for diagonal alignment
        if 30 <= abs(angle) <= 60 or 120 <= abs(angle) <= 150:
            rules.append(CompositionRule.DIAGONAL_LINES)

        # check for leading line potential
        start_x, start_y = start
        end_x, end_y = end

        # check if line starts near edge
        near_edge = (
                start_x < width * 0.1 or start_x > width * 0.9 or
                start_y < height * 0.1 or start_y > height * 0.9
        )

        # check if line points toward center
        center_x, center_y = width // 2, height // 2
        points_to_center = (
                abs(end_x - center_x) < abs(start_x - center_x) and
                abs(end_y - center_y) < abs(start_y - center_y)
        )

        if near_edge and points_to_center:
            rules.append(CompositionRule.LEADING_LINES)

        # check for symmetry potential
        if abs(angle) < 5 or abs(angle - 90) < 5:  # Horizontal or vertical
            rules.append(CompositionRule.SYMMETRY)

        return rules

    async def _calculate_rule_scores(
            self,
            frame: np.ndarray,
            subject_groups: List[Dict],
            elements: List[CompositionElement]
    ) -> Dict[CompositionRule, float]:
        """calculate scores for each composition rule"""
        height, width = frame.shape[:2]

        scores = {
            CompositionRule.RULE_OF_THIRDS: self._calculate_rule_of_thirds_score(
                subject_groups, height, width
            ),
            CompositionRule.GOLDEN_RATIO: self._calculate_golden_ratio_score(
                subject_groups, height, width
            ),
            CompositionRule.LEADING_LINES: self._calculate_leading_lines_score(
                elements
            ),
            CompositionRule.SYMMETRY: await self._calculate_symmetry_score(
                frame, subject_groups
            ),
            CompositionRule.DIAGONAL_LINES: self._calculate_diagonal_lines_score(
                elements
            ),
            CompositionRule.FRAMING: self._calculate_framing_score(
                elements, subject_groups
            ),
            CompositionRule.PATTERNS: self._calculate_patterns_score(elements)
        }

        return scores

    def _calculate_rule_of_thirds_score(
            self,
            subject_groups: List[Dict],
            height: int,
            width: int
    ) -> float:
        """calculate rule of thirds score """
        if not subject_groups:
            return 0.0

        # get rule of thirds
        intersection_points = calculate_rule_of_thirds_points(width, height)

        total_score = 0.0

        for group in subject_groups:
            # get group center in 3D
            center_3d = group.get('center_position', (0, 0, 5))

            # project to 2D
            center_x, center_y = self._project_3d_to_2d(center_3d, (height, width))

            # find nearest composition point
            nearest_point = find_nearest_composition_point(
                (center_x, center_y),
                intersection_points
            )

            # calculate distance
            distance = calculate_2d_distance(
                Point2D(center_x, center_y),
                Point2D(*nearest_point)
            )

            # score based on proximity to intersection points
            max_distance = np.sqrt(width ** 2 + height ** 2) * 0.2  # 20% of diagonal
            proximity_score = max(0, 1.0 - (distance / max_distance))
            total_score += proximity_score

        return min(1.0, total_score / len(subject_groups))

    def _calculate_golden_ratio_score(
            self,
            subject_groups: List[Dict],
            height: int,
            width: int
    ) -> float:
        """calculate golden ratio score"""
        if not subject_groups:
            return 0.0

        # get golden ratio points
        golden_points = calculate_golden_ratio_points(width, height)

        total_score = 0.0

        for group in subject_groups:
            center_3d = group.get('center_position', (0, 0, 5))

            # project to 2D
            center_x, center_y = self._project_3d_to_2d(center_3d, (height, width))

            # find nearest golden point
            nearest_point = find_nearest_composition_point(
                (center_x, center_y),
                golden_points
            )

            # calculate distance
            distance = calculate_2d_distance(
                Point2D(center_x, center_y),
                Point2D(*nearest_point)
            )

            max_distance = np.sqrt(width ** 2 + height ** 2) * 0.15
            proximity_score = max(0, 1.0 - (distance / max_distance))
            total_score += proximity_score

        return min(1.0, total_score / len(subject_groups))

    def _calculate_leading_lines_score(self, elements: List[CompositionElement]) -> float:
        """calculate leading lines score"""
        leading_line_elements = [
            e for e in elements
            if CompositionRule.LEADING_LINES in e.rule_alignment
        ]

        if not leading_line_elements:
            return 0.0

        # score based on number and strength of leading lines
        total_strength = sum(e.strength for e in leading_line_elements)

        # normalize by ideal number of leading lines (2-4)
        ideal_count = 3
        count_factor = min(1.0, len(leading_line_elements) / ideal_count)

        return min(1.0, (total_strength / len(leading_line_elements)) * count_factor)

    async def _calculate_symmetry_score(
            self,
            frame: np.ndarray,
            subject_groups: List[Dict]
    ) -> float:
        """calculate symmetry score"""
        # Use optimized symmetry detection
        vertical_symmetry = detect_symmetry(frame, axis="vertical")
        horizontal_symmetry = detect_symmetry(frame, axis="horizontal")

        # best symmetry score
        symmetry_score = max(vertical_symmetry, horizontal_symmetry)

        # bonus for symmetric subject placement
        height, width = frame.shape[:2]
        subject_symmetry_bonus = self._calculate_subject_symmetry_bonus(
            subject_groups, height, width
        )

        return min(1.0, symmetry_score + subject_symmetry_bonus)

    def _calculate_subject_symmetry_bonus(
            self,
            subject_groups: List[Dict],
            height: int,
            width: int
    ) -> float:
        """calculate bonus for symmetric subject placement."""
        if len(subject_groups) != 2:
            return 0.0

        # get positions of both subjects
        pos1 = subject_groups[0].get('center_position', (0, 0, 5))
        pos2 = subject_groups[1].get('center_position', (0, 0, 5))

        # check if subjects are symmetrically placed
        center_x, center_y = 0, 0  # World center

        # calculate distances from center
        dist1_x = abs(pos1[0] - center_x)
        dist1_y = abs(pos1[1] - center_y)
        dist2_x = abs(pos2[0] - center_x)
        dist2_y = abs(pos2[1] - center_y)

        # check symmetry
        symmetry_x = 1.0 - abs(dist1_x - dist2_x) / max(dist1_x + dist2_x, 1.0)
        symmetry_y = 1.0 - abs(dist1_y - dist2_y) / max(dist1_y + dist2_y, 1.0)

        return (symmetry_x + symmetry_y) * 0.15  # Bonus up to 0.3

    def _calculate_diagonal_lines_score(self, elements: List[CompositionElement]) -> float:
        """calculate score for diagonal lines in composition."""
        diagonal_elements = [
            e for e in elements
            if CompositionRule.DIAGONAL_LINES in e.rule_alignment
        ]

        if not diagonal_elements:
            return 0.0

        total_strength = sum(e.strength for e in diagonal_elements)
        avg_strength = total_strength / len(diagonal_elements)

        return min(1.0, avg_strength)

    def _calculate_framing_score(
            self,
            elements: List[CompositionElement],
            subject_groups: List[Dict]
    ) -> float:
        """calculate natural framing score."""
        framing_elements = [
            e for e in elements
            if CompositionRule.FRAMING in e.rule_alignment
        ]

        if not framing_elements or not subject_groups:
            return 0.0

        framing_strength = sum(e.strength for e in framing_elements)
        return min(1.0, framing_strength / 2.0)  # Normalize

    def _calculate_patterns_score(self, elements: List[CompositionElement]) -> float:
        """calculate score for repeating patterns."""
        pattern_elements = [
            e for e in elements
            if CompositionRule.PATTERNS in e.rule_alignment
        ]

        if not pattern_elements:
            return 0.0

        return min(1.0, len(pattern_elements) * 0.2)

    def _calculate_grid_intersections(self, height: int, width: int) -> List[Tuple[int, int]]:
        """calculate grid intersections points"""
        intersections = []

        # rule of thirds intersections
        intersections.extend(calculate_rule_of_thirds_points(width, height))

        # golden ratio intersections
        intersections.extend(calculate_golden_ratio_points(width, height))

        # center point
        intersections.append((width // 2, height // 2))

        return intersections

    async def _detect_leading_lines(self, frame: np.ndarray) -> List[List[Tuple[int, int]]]:
        """detect leading lines in the frame"""
        # Use optimized implementation from geometry_utils
        all_lines = find_leading_lines(
            frame,
            min_line_length=self.config.min_line_length,
            max_line_gap=self.config.max_line_gap
        )

        # filter for actual leading lines (point toward center)
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)

        leading_lines = []
        for line in all_lines:
            if len(line) == 2:
                angle = calculate_line_angle(line)
                if self._is_leading_line(line, center, angle):
                    leading_lines.append(line)

        return leading_lines

    def _is_leading_line(
            self,
            line: List[Tuple[int, int]],
            center: Tuple[int, int],
            angle: float
    ) -> bool:
        """check if a line is actually a leading line"""
        if len(line) != 2:
            return False

        start, end = line
        start_point = Point2D(*start)
        end_point = Point2D(*end)
        center_point = Point2D(*center)

        # check if end is closer to center than start
        start_dist = start_point.distance_to(center_point)
        end_dist = end_point.distance_to(center_point)

        return end_dist < start_dist

    async def _detect_symmetry_axes(
            self,
            frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """detect major symmetry axes in the frame."""
        height, width = frame.shape[:2]

        symmetry_axes = []

        # vertical center axis
        symmetry_axes.append(((width // 2, 0), (width // 2, height)))

        # horizontal center axis
        symmetry_axes.append(((0, height // 2), (width, height // 2)))

        return symmetry_axes

    def _calculate_overall_composition_score(
            self,
            rule_scores: Dict[CompositionRule, float]
    ) -> float:
        """calculate overall composition quality score."""
        # Weight different rules based on importance
        rule_weights = {
            CompositionRule.RULE_OF_THIRDS: 0.30,
            CompositionRule.GOLDEN_RATIO: 0.20,
            CompositionRule.LEADING_LINES: 0.20,
            CompositionRule.SYMMETRY: 0.15,
            CompositionRule.DIAGONAL_LINES: 0.10,
            CompositionRule.FRAMING: 0.10,
            CompositionRule.PATTERNS: 0.05
        }

        weighted_score = 0.0
        total_weight = 0.0

        for rule, score in rule_scores.items():
            weight = rule_weights.get(rule, 0.1)
            weighted_score += score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    async def _generate_composition_recommendations(
            self,
            rule_scores: Dict[CompositionRule, float],
            subject_groups: List[Dict],
            elements: List[CompositionElement]
    ) -> List[str]:
        """generate recommendations for improving composition."""
        recommendations = []

        # check for low-scoring rules and suggest improvements
        if rule_scores.get(CompositionRule.RULE_OF_THIRDS, 0) < 0.3:
            recommendations.append(
                "Consider positioning subjects along rule-of-thirds lines or intersections"
            )

        if rule_scores.get(CompositionRule.LEADING_LINES, 0) < 0.2:
            recommendations.append(
                "Look for leading lines to guide the viewer's eye to the subjects"
            )

        if rule_scores.get(CompositionRule.SYMMETRY, 0) < 0.3 and len(subject_groups) == 2:
            recommendations.append(
                "Consider symmetric positioning for the two subjects"
            )

        if not elements:
            recommendations.append(
                "The scene lacks strong compositional elements - consider changing angle"
            )

        # check subject positioning
        for i, group in enumerate(subject_groups):
            pos = group.get('center_position', (0, 0, 5))
            distance = pos[2]

            if distance > 6.0:
                recommendations.append(
                    f"Subject group {i + 1} is quite distant - consider moving closer"
                )
            elif distance < 1.5:
                recommendations.append(
                    f"Subject group {i + 1} is very close - consider stepping back"
                )

        return recommendations[:5]  # limit  == 5
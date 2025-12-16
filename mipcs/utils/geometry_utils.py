from pathlib import Path
import sys
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Point2D:
    """2D point representation"""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        """convert 2D point to tuple"""
        return self.x, self.y

    def to_int_tuple(self) -> Tuple[int, int]:
        """convert 2D point to int tuple"""
        return int(self.x), int(self.y)

    def distance_to(self, other: 'Point2D') -> float:
        """calculate distance to another 2D point"""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @classmethod
    def from_tuple(cls, point: Tuple[float, float]) -> 'Point2D':
        """create 2D point from tuple"""
        return cls(point[0], point[1])


@dataclass
class Point3D:
    """3D point representation"""
    x: float
    y: float
    z: float

    def to_tuple(self) -> Tuple[float, float, float]:
        """convert 3D point to tuple"""
        return self.x, self.y, self.z

    def distance_to(self, other: 'Point3D') -> float:
        """calculate distance to another 3D point"""
        return np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) **2
        )

    def horizontal_distance_to(self, other: 'Point3D') -> float:
        """calculate horizontal distance to another 3D point"""
        return np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2
        )

    def to_numpy(self) -> np.ndarray:
        """convert 3D point to numpy array"""
        return np.array([self.x, self.y, self.z])

    def to_homogeneous(self) -> np.ndarray:
        """Convert 3D point to homogeneous coords [x, y, z, 1]"""
        return np.array([self.x, self.y, self.z, 1.0])

    @classmethod
    def from_tuple(cls, point: Tuple[float, float, float]) -> 'Point3D':
        """create 3D point from tuple"""
        return cls(point[0], point[1], point[2])

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'Point3D':
        """create 3D point from numpy array"""
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    @classmethod
    def from_homogeneous(cls, arr: np.ndarray) -> 'Point3D':
        """convert homogeneous coords to 3D point"""
        if arr[3] != 0:
            return cls(float(arr[0] / arr[3]), float(arr[1] / arr[3]), float(arr[2] / arr[3]))
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))



@dataclass
class BoundingBox:
    """bounding box representation"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> int:
        return int(self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return int(self.y_max - self.y_min)

    @property
    def center(self) -> Point2D:
        return Point2D(
            (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2
        )

    @property
    def area(self) -> int:
        return self.width * self.height

def calculate_3d_distance(p1: Point3D, p2: Point3D) -> float:
    """calculate Euclidean distance between two 3d points"""
    return p1.distance_to(p2)

def calculate_2d_distance(p1: Point2D, p2: Point2D) -> float:
    """calculate Euclidean distance between two 2D points"""
    return p1.distance_to(p2)

def calculate_horizontal_distance(p1: Point3D, p2: Point3D) -> float:
    """calculate horizontal distance between two 3d points"""
    return p1.horizontal_distance_to(p2)

def calculate_angle_between_points(p1: Point3D, p2: Point3D, p3: Point3D) -> float:

    #vectors from p1, p2 & p3
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    #cosine angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))

def calculate_yaw_pitch_to_point(camera_pos: Point3D, target_pos: Point3D, coordinate_frame: str = "base_link") -> Tuple[float, float]:
    """calculate yaw and pitch angles from camera to target point"""

    dx = target_pos.x - camera_pos.x
    dy = target_pos.y - camera_pos.y
    dz = target_pos.z - camera_pos.z

    if coordinate_frame == "base_link":
        # Robot frame: X=forward | Y = left | Z = up
        # Standard yaw calculation
        yaw_standard = np.arctan2(dy, dx) * 180.0 / np.pi

        # MANRIIX (inverted gimbal axis (0 0 -1)) --> negative --> Check urdf
        yaw = -yaw_standard

        #Pitch --> UP (+)
        horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
        if horizontal_dist < 1e-6:
            pitch = 0.0
        else:
            pitch = np.arctan2(dz, horizontal_dist) * 180.0 / np.pi

    elif coordinate_frame == "camera_optical":
        # Camera optical frame: X=right, Y=down, Z=forward
        # Transform to base_link frame first
        base_x = dz  # Camera Z (forward) → Base X (forward)
        base_y = -dx  # Camera X (right) → Base -Y (right = -left)
        base_z = -dy  # Camera Y (down) → Base -Z (down = -up)

        # calculate yaw/pitch in base frame
        yaw_standard = np.arctan2(base_y, base_x) * 180.0 / np.pi

        # MANRIIX (inverted gimbal axis (0 0 -1)) --> negative --> Check urdf
        yaw = -yaw_standard

        horizontal_dist = np.sqrt(base_x ** 2 + base_y ** 2)
        if horizontal_dist < 1e-6:
            pitch = 0.0
        else:
            pitch = np.arctan2(base_z, horizontal_dist) * 180.0 / np.pi
    else:
        logger.warning(f"Unknown coordinate frame: {coordinate_frame}. [DEFAULT: {coordinate_frame}]")
        yaw = 0.0
        pitch = 0.0

    return yaw, pitch


def calculate_bounding_box(points: List[Tuple[float, float]]) -> BoundingBox:
    """calculate bounding box from list of 2D points"""

    if not points:
        return BoundingBox(0, 0, 0, 0)

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    return BoundingBox(
        x_min=float(min(x_coords)),
        y_min=float(min(y_coords)),
        x_max=float(max(x_coords)),
        y_max=float(max(y_coords))
    )


def merge_bounding_boxes(boxes: List[BoundingBox]) -> BoundingBox:
    """merge multiple bounding boxes into one"""

    if not boxes:
        return BoundingBox(0, 0, 0, 0)

    x_min = min(box.x_min for box in boxes)
    y_min = min(box.y_min for box in boxes)
    x_max = max(box.x_max for box in boxes)
    y_max = max(box.y_max for box in boxes)

    return BoundingBox(x_min, y_min, x_max, y_max)

def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """calculate IoU between two bounding boxes"""

    #cal intersection
    x_left = max(box1.x_min, box2.x_min)
    y_top = max(box1.y_min, box2.y_min)
    x_right = min(box1.x_max, box2.x_max)
    y_bottom = min(box1.y_max, box2.y_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    #cal union
    box1_area = box1.area
    box2_area = box2.area
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def expand_bounding_box(
        box: BoundingBox,
        padding_ratio: float,
        image_width: int,
        image_height: int
) -> BoundingBox:
    """Expand bounding box by padding ratio, clamped to image boundries"""

    width_padding = int(box.width * padding_ratio)
    height_padding = int(box.height * padding_ratio)

    x_min = max(0, box.x_min - width_padding)
    y_min = max(0, box.y_min - height_padding)
    x_max = min(image_width, box.x_max + width_padding)
    y_max = min(image_height, box.y_max + height_padding)

    return BoundingBox(x_min, y_min, x_max, y_max)


def optimize_frame_coverage(
        subject_positions: List[Point3D],
        frame_width: int,
        frame_height: int,
        padding_ratio: float = 0.15
) -> Dict[str, float]:
    """calculate optimal frame coverage for subjects"""

    if not subject_positions:
        return {'zoom': 1.0, 'pan': 0.0, 'tilt': 0.0, 'spread_x': 0.0, 'spread_y': 0.0}

    # cal bboxes of subjects
    x_coords = [p.x for p in subject_positions]
    y_coords = [p.y for p in subject_positions]
    z_coords = [p.z for p in subject_positions]

    # subject spread
    x_spread = max(x_coords) - min(x_coords)
    y_spread = max(y_coords) - min(y_coords)

    # cal required zoom to fit subjects with padding
    required_h_zoom = x_spread * (1 + padding_ratio)
    required_v_zoom = y_spread * (1 + padding_ratio)

    # use larger zoom  requirement
    optimal_zoom = max(required_h_zoom, required_v_zoom, 1.0)

    # cal centroid for pan/tilt
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)

    return {
        'zoom': optimal_zoom,
        'pan': centroid_x,
        'tilt': centroid_y,
        'spread_x': x_spread,
        'spread_y': y_spread
    }

def calculate_rule_of_thirds_points(
        frame_width: int,
        frame_height: int
) -> List[Tuple[int, int]]:
    """calculate rule of third intersection points"""

    x_lines = [frame_width // 3, 2 * frame_width // 3]
    y_lines = [frame_height // 3, 2 * frame_height // 3]

    #cal all 4 intersection points
    intersections = []
    for x in x_lines:
        for y in y_lines:
            intersections.append((x, y))

    return intersections

def calculate_golden_ratio_points(
        frame_width: int,
        frame_height: int
) -> List[Tuple[int, int]]:
    """calculate golden ratio intersection points"""

    x_lines = [int(frame_width * 0.382), int(frame_width * 0.618)]
    y_lines = [int(frame_height * 0.382), int(frame_height * 0.618)]

    intersections = []
    for x in x_lines:
        for y in y_lines:
            intersections.append((x, y))

    return intersections


def find_nearest_composition_point(
        point: Tuple[int, int],
        composition_points: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """find nearest composition point (rule of thirds or golden ratio) to given point"""

    if not composition_points:
        return point

    distances = [
        np.sqrt((point[0] - cp[0]) ** 2 + (point[1] - cp[1]) ** 2)
        for cp in composition_points
    ]
    nearest_idx = np.argmin(distances)
    return composition_points[nearest_idx]


###

def calculate_centroid_2d(
        points: List[Point2D]
) -> Point2D:
    """calculate centroid 2d point"""
    if not points:
        return Point2D(0.0, 0.0)

    x_mean = np.mean([p.x for p in points])
    y_mean = np.mean([p.y for p in points])

    return Point2D(x_mean, y_mean)

def calculate_centroid_3d(
        points: List[Point3D]
) -> Point3D:
    """calculate centroid 3d point"""
    if not points:
        return Point3D(0.0, 0.0, 0.0)

    x_mean = np.mean([p.x for p in points])
    y_mean = np.mean([p.y for p in points])
    z_mean = np.mean([p.z for p in points])

    return Point3D(x_mean, y_mean, z_mean)

def calculate_weighted_centroid_3d(
    points: List[Point3D],
    weights: List[float]
) -> Point3D:
    """calculate weighted centroid 3d point"""
    if not points or not weights or len(points) != len(weights):
        return Point3D(0.0, 0.0, 0.0)

    total_weight = sum(weights)
    if total_weight == 0:
        return calculate_centroid_3d(points)

    x_weighted = sum(p.x * w for p, w in zip(points, weights)) / total_weight
    y_weighted = sum(p.y * w for p, w in zip(points, weights)) / total_weight
    z_weighted = sum(p.z * w for p, w in zip(points, weights)) / total_weight

    return Point3D(x_weighted, y_weighted, z_weighted)

def find_leading_lines(
        frame: np.ndarray,
        min_line_length: int = 100,
        max_line_gap: int = 10
) -> List[List[Tuple[int, int]]]:
    """Detect leading lines in frame using Hough transform"""

    #convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return []

    # convert to list of point pairs
    detected_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        detected_lines.append([(x1, y1), (x2, y2)])

    return detected_lines


def calculate_line_angle(
        line: List[Tuple[int, int]]
) -> float:
    """calculate line angle in degrees"""

    if len(line) != 2:
        return 0.0

    p1, p2 = line

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle = np.arctan2(dy, dx) * 180 / np.pi

    # normalize (0 - 180)
    if angle < 0:
        angle += 180

    return angle


def detect_symmetry(
        frame: np.ndarray,
        axis: str = "vertical"
) -> float:
    """detect symmetry in frame along specified axis"""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if axis == "vertical":
        # split the frame vertically
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]

        # flip the right half
        right_half_flipped = cv2.flip(right_half, 1)

        # resize to match if needed
        if left_half.shape != right_half_flipped.shape:
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]

    elif axis == "horizontal":
        # split the frame horizontally
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]

        # bottom half flipped
        bottom_half_flipped = cv2.flip(bottom_half, 0)

        # resize to match if needed
        if top_half.shape != bottom_half_flipped.shape:
            min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
            left_half = top_half[:min_height, :]
            right_half_flipped = bottom_half_flipped[:min_height, :]
    else:
        return 0.0

    # calculate similarity ( based on correlation)
    corr = np.corrcoef(left_half.flatten(), right_half_flipped.flatten())[0, 1]

    # nom to 0 - 1
    symmetry_score = (corr + 1) / 2
    symmetry_score = np.clip(symmetry_score, 0.0, 1.0)

    return symmetry_score



def transform_point_with_matrix(
        point: Point3D,
        transformation_matrix: np.ndarray
) -> Point3D:
    """transform point with transformation matrix"""

    p_h = point.to_homogeneous()
    p_transformed_h = transformation_matrix @ p_h
    return Point3D.from_homogeneous(p_transformed_h)

def create_rotation_matrix_z(
        angle_rad: float
) -> np.ndarray:
    """create rotation matrix around z axis (yaw)"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def create_rotation_matrix_y(
        angle_rad: float
) -> np.ndarray:
    """create rotation matrix around y axis (pitch)"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def create_rotation_matrix_x(
        angle_rad: float
) -> np.ndarray:
    """create rotation matrix around x axis (roll)"""

    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def clamp_point_to_bounds(
        point: Point3D,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float]
) -> Point3D:
    """clamp 3D point to bounds"""

    return Point3D(
        x=np.clip(point.x, x_range[0], x_range[1]),
        y=np.clip(point.y, y_range[0], y_range[1]),
        z=np.clip(point.z, z_range[0], z_range[1])
    )

def points_to_numpy_array(points: List[Point3D]) -> np.ndarray:
    """convert list of Point3D to numpy array (Nx3)"""
    return np.array([[p.x, p.y, p.z] for p in points])


def numpy_array_to_points(arr: np.ndarray) -> List[Point3D]:
    """convert numpy array (Nx3) to list of Point3D"""
    return [Point3D(float(row[0]), float(row[1]), float(row[2])) for row in arr]




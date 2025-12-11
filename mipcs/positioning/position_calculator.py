import asyncio
import time
from pathlib import Path
import sys
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..vision.detector import PersonDetection
from ..vision.depth_processor import DepthProcessor, DepthData
from config.settings import get_settings
from ..utils.logger import get_logger, log_performance
from ..utils.exceptions import PositionCalculationError, DepthInvalidError

logger = get_logger(__name__)

class CoordinateFrame(Enum):
    """Coordinate frame systems"""
    CAMERA = auto()
    ROBOT_BASE = auto()
    WORLD = auto()

@dataclass
class PersonPosition:
    """3D position of a detected person"""

    #position data
    position_3d: Tuple[float, float, float]         # (x, y, z) in camera frame
    distance_from_camera: float                  # Euclidean distance from camera
    height_from_ground: float                     # Height relative to ground plane

    #detection data
    detection: PersonDetection        # Original detection data
    depth_result: Optional[DepthData]  # Extracted depth data

    #quality metrics
    position_confidence: float          # Confidence score of position estimation ( 0 - 1 )
    depth_quality: float               # Quality of depth data ( 0 - 1 )

    #derived properties
    azimuth_angle: float               # Angle in horizontal plane from camera center
    elevation_angle: float             # Angle in vertical plane from camera center

    #filtering and tracking
    is_filtered: bool = False                  # Whether the position has been filtered
    tracking_id: Optional[int] = None          # ID for tracking across frames
    velocity_3d: Optional[Tuple[float, float, float]] = None  # Velocity vector in 3D space

    #temporal data
    timestamp: float = 0.0            # position calculation timestamp
    frame_id: Optional[str] = None          # coordinate frame identifier

@dataclass
class PositionFilterState:
    """State for position filtering and tracking"""

    previous_positions: List[Tuple[float, float, float]]  #position history
    previous_timestamps: List[float]                        # timestamp history
    velocity_estimate: Tuple[float, float, float]      # current velocity estimate
    last_update: float                           #last filter update time

class PositionCalculator:
    """Position calculator"""

    def __init__(self, depth_processor: DepthProcessor):

        self.settings = get_settings()
        self.config = self.settings.positioning
        self.depth_processor = depth_processor

        #cam configs
        self.camera_height = self.config.camera_mounting.height_from_ground
        self.camera_tilt_rad = np.radians(self.config.camera_mounting.tilt_angle_deg)

        #position filtering
        self.filtering_enabled = self.config.positioning_filtering.enabled
        self.filter_states: Dict[int, PositionFilterState] = {}

        #performance tracking
        self.processing_stats = {
            'positions_calculated': 0,
            'filtering_applied': 0,
            'total_processing_time': 0.0,
            'failed_calculations': 0
        }

        #ground plane params
        self._init_ground_plane()

        logger.info("position_calculator_initialized",
                    camera_height=self.camera_height,
                    filtering_enabled=self.filtering_enabled,
                    coordinate_system=self.config.coordinate_system)

    def _init_ground_plane(self) -> None:
        """Initialize ground plane detection/configs"""
        ground_config = self.config.ground_plane

        if ground_config.detection_method == 'fixed_height':
            self.ground_plane_method = 'fixed'

        elif ground_config.detection_method == 'ransac':
            self.ground_plane_method = 'ransac'
            logger.warning("ransac_ground_detection_not_implemented_using_fixed")
            self.ground_plane_method = "fixed"   #TODO: IMPLEMENT RANSAC --later
        else:
            self.ground_plane_method = "fixed"
        logger.debug("ground_plane_initialized", method=self.ground_plane_method)

    @log_performance("position_calculation")
    async def calculate_positions_batch(
            self,
            person_detections: List[PersonDetection],
    ) -> List[Optional[PersonPosition]]:
        """Calculate 3D positions of detected persons"""
        start_time = time.time()

        if not person_detections:
            logger.debug("no_persons_to_process")
            return []

        try:
            bboxes = [detection.bbox for detection in person_detections]

            depth_results = await self.depth_processor.extract_multiple_depths(bboxes)

            positions = []
            for i, (detection, depth_result) in enumerate(zip(person_detections, depth_results)):
                try:
                    if depth_result is not None:
                        # calculate position
                        position = await self._calculate_single_position(detection, depth_result)
                        positions.append(position)
                    else:
                        # No depth data - try fallback estimation
                        fallback_position = self._estimate_position_fallback(detection)
                        positions.append(fallback_position)

                except Exception as e:
                    logger.warning("position_calculation_failed", person_id=i, error=str(e))
                    positions.append(None)
                    self.processing_stats['failed_calculations'] += 1

            #apply filtering
            if self.filtering_enabled:
                positions = await self._apply_position_filtering(positions)

            #update performance stats
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_stats(processing_time, len(person_detections))

            logger.debug("batch_position_calculation_complete",
                input_detections=len(person_detections),
                successful_positions=sum(1 for p in positions if p is not None),
                processing_time_ms=processing_time)

            return positions

        except Exception as e:
            logger.error("batch_position_calculation_error", error=str(e))
            return [None] * len(person_detections)

    async def _calculate_single_position(
            self,
            detection: PersonDetection,
            depth_result: DepthData
    ) -> PersonPosition:
        """Calculate 3D position of detected person"""

        x_cam, y_cam, z_cam = depth_result.position_3d

        #tilt correction if needed
        if abs(self.camera_tilt_rad) > 0.001: # >0.1 degree
            x_corrected, y_corrected, z_corrected = self._apply_tilt_correction(
                x_cam, y_cam, z_cam
            )
        else:
            x_corrected, y_corrected, z_corrected = x_cam, y_cam, z_cam


        #height from ground
        height_from_ground = self._calculate_ground_height(y_corrected)

        #cal angles
        azimuth = np.arctan2(x_corrected, z_corrected)
        elevation = np.arctan2(-y_corrected, z_corrected)   # -Y bcz y+ down

        #calculate distance
        distance = np.sqrt(x_corrected ** 2 + y_corrected ** 2 + z_corrected ** 2)

        #calculate confidence
        position_confidence = self._calculate_position_confidence(detection, depth_result)


        position = PersonPosition(
            position_3d=(x_corrected, y_corrected, z_corrected),
            distance_from_camera=distance,
            height_from_ground=height_from_ground,
            detection=detection,
            depth_result=depth_result,
            position_confidence=position_confidence,
            depth_quality=depth_result.quality_score,
            azimuth_angle=azimuth,
            elevation_angle=elevation,
            timestamp=time.time(),
            frame_id="camera_frame"
        )

        self.processing_stats['positions_calculated'] += 1

        return position


    def _apply_tilt_correction(
            self,
            x: float,
            y: float,
            z: float
    )-> Tuple[float, float, float]:
        """Apply tilt correction to 3D coords"""

        cos_tilt = np.cos(self.camera_tilt_rad)
        sin_tilt = np.sin(self.camera_tilt_rad)

        #apply rotation
        x_corrected = x
        y_corrected = y * cos_tilt - z * sin_tilt
        z_corrected = z * sin_tilt + z * cos_tilt

        return (x_corrected, y_corrected, z_corrected)

    def _calculate_ground_height(self, y_camera: float) -> float:
        """Calculate ground height"""

        if self.ground_plane_method == 'fixed':

            # Camera Y+ (down) --> ground_height = camera_height - y_camera
            height_from_ground = self.camera_height - y_camera

            #clamp
            height_from_ground = max(-0.5, min(3.0, height_from_ground))

            return height_from_ground
        else:
            return self.camera_height - y_camera   # TODO: RANSAC --> Later

    def _calculate_position_confidence(
            self,
            detection: PersonDetection,
            depth_result: DepthData
    ) -> float:
        """Calculate position confidence"""
        detection_conf = detection.confidence

        depth_conf = depth_result.quality_score

        distance_m= depth_result.depth_value
        if distance_m< 3.0:
            distance_conf = 1.0
        elif distance_m< 6.0:
            distance_conf = 1.0 - (distance_m- 3.0) / 3.0    # Linear decline from 3 to 6
        else:
            distance_conf = 0.2   #low conf beyond 6m

        #bbox size
        bbox = detection.bbox
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        size_conf = min(1.0, bbox_area / 50000.0)  #normalize by large bbox size

        #weighted combination
        confidence = (
            detection_conf * 0.3 +  # 30% detection confidence
            depth_conf * 0.4 +  # 40% depth quality
            distance_conf * 0.2 +  # 20% distance reliability
            size_conf * 0.1  # 10% size facto
        )

        return min(1.0, max(0.0, confidence))

    def _estimate_position_fallback(
            self,
            detection: PersonDetection,
    ) -> Optional[PersonPosition]:
        """fallback position estimation (when no depth data)"""

        global estimated_focal_length
        try:
            bbox = detection.bbox
            bbox_height = bbox[3] - bbox[1]

            #distance estimation assuming human height ~ 1.7 m
            if bbox_height > 0:
                # assuming camera with ~600px focal length for 640px image
                estimated_focal_length = 600.0
                estimated_distance = (1.7 * estimated_focal_length) / bbox_height

                # clamp
                estimated_distance = np.clip(estimated_distance, 2.0, 8.0)
            else:
                estimated_distance = 4.0  # Default fallback

                # Calculate approximate 3D position
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Rough deprojection (assumes image center at 320, 240)
            x_3d = (center_x - 320) * estimated_distance / estimated_focal_length
            y_3d = (center_y - 240) * estimated_distance / estimated_focal_length
            z_3d = estimated_distance

            # fallback position
            position = PersonPosition(
                position_3d=(x_3d, y_3d, z_3d),
                distance_from_camera=estimated_distance,
                height_from_ground=self.camera_height - y_3d,
                detection=detection,
                depth_result=None,
                position_confidence=0.3,  # Low confidence bcz estimation
                depth_quality=0.0,
                azimuth_angle=np.arctan2(x_3d, z_3d),
                elevation_angle=np.arctan2(-y_3d, z_3d),
                timestamp=time.time(),
                frame_id="camera_frame"
            )

            logger.warning("using_fallback_position_estimation", bbox=bbox, estimated_distance=estimated_distance)

            return position

        except Exception as e:
            logger.error("fallback_position_estimation_failed", error=str(e))
            return None

    async def _apply_position_filtering(
            self,
            positions: List[Optional[PersonPosition]],
    ) -> List[Optional[PersonPosition]]:
        """Apply position filtering"""

        if not self.filtering_enabled:
            return positions

        filtered_positions = []

        for i, position in enumerate(positions):
            if position is None:
                filtered_positions.append(None)
                continue

            try:
                #tracking id
                tracking_id = self._get_tracking_id(position, i)
                position.tracking_id = tracking_id

                #apply filtering
                filtered_position = self._filter_single_position(position)
                filtered_position.is_filtered = True
                filtered_positions.append(filtered_position)
                self.processing_stats['filtering_applied'] += 1

            except Exception as e:
                logger.warning("position_filtering_failed", error=str(e))
                filtered_positions.append(position)

        return filtered_positions

    def _get_tracking_id(
            self,
            position: PersonPosition,
            detection_index: int
    )-> int:
        """Get tracking id"""

        current_pos = position.position_3d
        min_distance = float("inf")
        best_id = None

        for track_id, filter_state in self.filter_states.items():
            if filter_state.previous_positions:
                last_pos = filter_state.previous_positions[-1]
                distance = np.sqrt(sum((a - b)**2 for a, b in zip(current_pos, last_pos)))

                if distance < min_distance and distance < 0.5:  #threshold = 50cm
                    min_distance = distance
                    best_id = track_id

        if best_id is None:
            #new id
            best_id = len(self.filter_states) + detection_index

        return best_id

    def _filter_single_position(self, position: PersonPosition) -> PersonPosition:
        """Filter single position"""

        track_id = position.tracking_id
        current_time = position.timestamp
        current_pos = position.position_3d

        #get or create filter state
        if track_id not in self.filter_states:
            self.filter_states[track_id] = PositionFilterState(
                previous_positions=[],
                previous_timestamps=[],
                velocity_estimate=(0.0, 0.0, 0.0),
                last_update=current_time
            )

        filter_state = self.filter_states[track_id]

        #median filtering
        if len(filter_state.previous_positions) >= self.config.positioning_filtering.median_window_size:

            #remove oldest positions
            filter_state.previous_positions.pop(0)
            filter_state.previous_timestamps.pop(0)

        #add current position
        filter_state.previous_positions.append(current_pos)
        filter_state.previous_timestamps.append(current_time)

        #calculate filtered position using median filter
        if len(filter_state.previous_positions) >= 3:

            #apply median filtering across recent positions
            positions_array = np.array(filter_state.previous_positions)
            filtered_pos = tuple(np.median(positions_array, axis=0))

            # Velocity estimation
            if len(filter_state.previous_positions) >= 2:
                dt = filter_state.previous_timestamps[-1] - filter_state.previous_timestamps[-2]
                if dt > 0:
                    velocity = tuple((filtered_pos[i] - filter_state.previous_positions[-2][i]) / dt for i in range(3))
                    filter_state.velocity_estimate = velocity
                    position.velocity_3d = velocity
        else:
            filtered_pos = current_pos

            # Update filter state
            filter_state.last_update = current_time

            # Create new position with filtered coordinates
            filtered_position = PersonPosition(
                position_3d=filtered_pos,
                distance_from_camera=np.sqrt(sum(x ** 2 for x in filtered_pos)),
                height_from_ground=self.camera_height - filtered_pos[1],
                detection=position.detection,
                depth_result=position.depth_result,
                position_confidence=position.position_confidence,
                depth_quality=position.depth_quality,
                azimuth_angle=np.arctan2(filtered_pos[0], filtered_pos[2]),
                elevation_angle=np.arctan2(-filtered_pos[1], filtered_pos[2]),
                tracking_id=track_id,
                velocity_3d=position.velocity_3d,
                timestamp=current_time,
                frame_id=position.frame_id
            )

            return filtered_position

    def _update_processing_stats(
            self,
            processing_time_ms: float,
            num_detections: int
    ) -> None:
        """Update processing performance statistics"""
        self.processing_stats['total_processing_time'] += processing_time_ms

        #cal average processing time
        total_calculations = self.processing_stats['positions_calculated']
        if total_calculations > 0:
            avg_time = self.processing_stats['total_processing_time'] / total_calculations
            self.processing_stats['average_processing_time_ms'] = avg_time

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            'active_tracks': len(self.filter_states),
            'filtering_enabled': self.filtering_enabled,
            'ground_plane_method': self.ground_plane_method
        }

    def reset_tracking(self) -> None:
        """Reset all tracking states"""
        self.filter_states.clear()
        logger.info("position_tracking_reset")


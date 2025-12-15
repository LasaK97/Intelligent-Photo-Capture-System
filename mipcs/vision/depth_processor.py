import asyncio
import time
from email.mime import image
from time import process_time

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import cv2

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = None
    Image = None
    CameraInfo = None
    CvBridge = None

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from ..utils.logger import get_logger, log_performance
from ..utils.threading_utils import get_thread_manager, async_retry
from ..utils.exceptions import (
    DepthProcessingError,
    DepthDataError,
    CalibrationError,
    TimeoutError
)

logger = get_logger(__name__)


class DepthData:
    """data class for depth processing results"""

    def __init__(
            self,
            depth_value: float,
            confidence: float,
            position_3d: Tuple[float, float, float],
            roi_stats: Dict[str, float],
            processing_time_ms: float,
            quality_score: float = 0.0,
            extraction_method: str = "center_roi",
            depth_quality_reason: str = "unknown",
    ):
        self.depth_value = depth_value
        self.confidence = confidence
        self.position_3d = position_3d
        self.roi_stats = roi_stats
        self.processing_time_ms = processing_time_ms,
        self.quality_score = quality_score,
        self.extraction_method = extraction_method,
        self.depth_quality_reason = depth_quality_reason,

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'depth_value': self.depth_value,
            'confidence': self.confidence,
            'position_3d': self.position_3d,
            'roi_stats': self.roi_stats,
            'processing_time_ms': self.processing_time_ms,
            'quality_score': self.quality_score,
            'extraction_method': self.extraction_method,
            'depth_quality_reason': self.depth_quality_reason,
        }


class CameraIntrinsics:
    """Intrinsic parameters for depth projection."""

    def __init__(self, camera_info: Optional[Any] = None):
        if camera_info is not None:
            self.fx = camera_info.k[0]
            self.fy = camera_info.k[4]
            self.cx = camera_info.k[2]
            self.cy = camera_info.k[5]
            self.width = camera_info.width
            self.height = camera_info.height
        else:
            # default reslasense d455 intrinsics
            self.fx = 640.0
            self.fy = 640.0
            self.cx = 640.0
            self.cy = 360.0
            self.width = 1280
            self.height = 720

    def deproject_pixel_to_point(
            self,
            pixel: Tuple[int, int],
            depth: float,
    ) -> Tuple[float, float, float]:
        """Convert 2D pixel + depth to 3D point in camera frame"""

        u, v = pixel
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        return (x, y, z)

    def project_point_to_pixel(
            self,
            point: Tuple[float, float, float]
    ) -> Tuple[int, int]:
        """Convert 3D point to pixel coordinate"""

        x, y, z = point

        if z <= 0:
            raise ValueError("Invalid depth value for projection.( Z must be greater than 0)")

        u = int((x * self.fx) / z + self.cx)
        v = int((y * self.fy) / z + self.cy)

        return (u, v)

    def is_valid_pixel(self, pixel: Tuple[int, int]) -> bool:
        """Check if pixel is valid"""
        u, v = pixel
        return 0 <= u < self.width and 0 <= v < self.height


##quality checking class
class DepthQualityChecker:
    """validates depth data quality """

    def __init__(self, config):
        self.config = config
        self.min_valid_ratio = config.min_valid_pixels_ratio
        self.depth_range = (config.depth_range.min, config.depth_range.max)

    def check_depth_quality(
            self,
            depth_roi: np.ndarray
    ) -> Tuple[bool, float, str]:
        if depth_roi is None or depth_roi.size == 0:
            return False, 0, "empty_roi"

        # check valid pixel percentage
        min_depth_mm = self.depth_range[0] * 1000
        max_depth_mm = self.depth_range[1] * 1000
        valid_mask = (depth_roi > min_depth_mm) & (depth_roi < max_depth_mm)
        valid_ratio = np.count_nonzero(valid_mask) / depth_roi.size

        if valid_ratio < self.min_valid_ratio:
            return False, valid_ratio, "insufficient_valid_pixels"

        # check for noise patterns
        valid_depths = depth_roi[valid_mask]
        if len(valid_depths) < 5:
            return False, valid_ratio, "too_few_valid_points"

        # calculate quality score (0.0 - 1.0)
        depth_std = np.std(valid_depths)
        depth_mean = np.mean(valid_depths)

        # lower coef of variation = higher quality
        cv = depth_std / depth_mean if depth_mean > 0 else 1.0
        quality_score = valid_ratio * (1.0 - min(cv, 0.5))

        return True, quality_score, "valid"

    def detect_depth_holes(self, depth_roi: np.ndarray) -> bool:
        """Detect if ROI has significant depth holes (missing data)"""
        if depth_roi is None or depth_roi.size == 0:
            return True

        zero_mask = depth_roi == 0
        zero_ratio = np.count_nonzero(zero_mask) / depth_roi.size

        # if more than 40% holes --> PROBLEM
        return zero_ratio > 0.4


class DepthProcessor:

    def __init__(self, node: Optional[Node] = None):
        self.settings = get_settings()
        self.config = self.settings.vision.realsense
        self.node = node

        # camera calibration
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.intrinsics_received = False

        # current depth data
        self.latest_depth_image: Optional[np.ndarray] = None
        self.depth_timestamp = 0.0

        # ROS components
        self.bridge = CvBridge() if ROS_AVAILABLE else None
        self.depth_subscription = None
        self.camera_info_subscription = None

        # performance tracking
        self.processing_count = 0
        self.total_processing_time = 0
        self.processing_stats = {
            'frames_processed': 0,
            'failed_extractions': 0,
            'average_processing_time_ms': 0.0,
            'extraction_method_counts': {
                'center_roi': 0,
                'expanded_roi': 0,
                'statistical': 0,
                'fallback_estimate': 0
            }
        }

        self.thread_manager = get_thread_manager()

        # init quality checker
        self.quality_checker = DepthQualityChecker(self.config)

        if self.node and ROS_AVAILABLE:
            self._setup_ros_subscription()

        logger.info("depth_processor_initialized")

    def _setup_ros_subscription(self) -> None:
        """Set up ROS subscription for depth processing."""
        if not self.node or not ROS_AVAILABLE:
            return

        self.depth_subscription = self.node.create_subscription(
            Image,
            self.settings.ros2_topics.realsense_depth,
            self._depth_callback,
            10
        )

        self.camera_info_subscription = self.node.create_subscription(
            CameraInfo,
            self.settings.ros2_topics.realsense_camera_info,
            self._camera_info_callback,
            10
        )

        logger.info(
            "depth_ros_subscriptions_created",
            depth_topic=self.settings.ros2_topics.realsense_depth,
            camera_info_topic=self.settings.ros2_topics.realsense_camera_info
        )

    def _depth_callback(self, msg: Image) -> None:
        """ROS Callback for depth image."""
        try:
            # convert ros image to numpy array
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")

            if not self._validate_depth_image(depth_image):  ##### CHECK
                logger.warning("invalid_depth_image_received")
                return

            self.latest_depth_image = depth_image
            self.depth_timestamp = time.time()

            # update processing stats
            self.processing_stats['frames_processed'] += 1

            logger.debug("depth_image_received", shape=depth_image.shape, encoding=msg.encoding)
        except Exception as e:
            logger.error("depth_callback_error", error=str(e))

    def _validate_depth_image(self, depth_image: np.ndarray) -> bool:
        """validate depth image data"""
        if depth_image is None:
            return False

        if depth_image.size == 0:
            return False

        if depth_image.dtype != np.uint16:
            return False

        return True


    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """ROS Callback for camera info."""
        try:
            self.intrinsics = CameraIntrinsics(msg)
            self.intrinsics_received = True

            logger.info(
                "camera_intrinsics_received",
                fx=self.intrinsics.fx,
                fy=self.intrinsics.fy,
                cx=self.intrinsics.cx,
                cy=self.intrinsics.cy,
                resolution=(self.intrinsics.width, self.intrinsics.height)
            )

        except Exception as e:
            logger.error("camera_info_callback_error", error=str(e))

    async def wait_for_calibration(self, timeout: float = 10.0) -> bool:
        """Wait for camera intrinsics to be received."""
        start_time = time.time()

        while not self.intrinsics_received and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        if not self.intrinsics_received:
            logger.warning("camera_calibration_timeout")
            return False

        logger.info("camera_calibration_ready")
        return True

    @log_performance("depth_extraction")
    async def extract_person_depth(
            self,
            bbox: Tuple[int, int, int, int],
            depth_image: Optional[np.ndarray] = None,
    ) -> Optional[DepthData]:
        """Extract robust depth value for person bbox with multi-level fallbacks"""

        start_time = time.time()

        if depth_image is None:
            depth_image = self.latest_depth_image

        if depth_image is None:
            logger.warning("no_depth_image_received")
            return None

        # validate depth image
        if not self._validate_depth_image(depth_image):
            logger.warning("invalid_depth_image_received")
            return None

        # check intrinsics
        if not self.intrinsics_received:
            logger.warning("camera_intrinsics_not_available")
            self.intrinsics = CameraIntrinsics()
            self.intrinsics_received = True

        try:
            result = await self._extract_depth_multi_level(bbox, depth_image)

            if result is None:
                logger.debug("all_depth_extraction_methods_failed", bbox=bbox)
                return None

            depth_value, confidence, roi_stats, method_used = result

            # calculate 3D position
            bbox_center = (
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            )

            position_3d = self.intrinsics.deproject_pixel_to_point(
                bbox_center,
                depth_value
            )

            # cal processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # update performance metrics
            self._update_performance_metrics(processing_time_ms)

            depth_data = DepthData(
                depth_value=depth_value,
                confidence=confidence,
                position_3d=position_3d,
                roi_stats=roi_stats,
                processing_time_ms=processing_time_ms,
                quality_score=roi_stats.get('quality_score', 0.0),
                extraction_method=method_used,
                depth_quality_reason="valid"
            )

            logger.debug(
                "depth_extraction_successful",
                bbox=bbox,
                depth=depth_value,
                confidence=confidence,
                method=method_used,
                processing_time_ms=processing_time_ms
            )

            return depth_data

        except Exception as e:
            logger.error("depth_extraction_error", error=str(e), bbox=bbox)
            raise DepthProcessingError(f"Depth extraction failed: {e}") from e

    def _extract_depth_sync(
            self,
            bbox: Tuple[int, int, int, int],
            depth_image: np.ndarray,
    ) -> Optional[Tuple[float, float, Dict[str, float]]]:
        """Synchronously extract depth value from depth image"""
        x1, y1, x2, y2 = bbox

        # validate bbox
        h, w = depth_image.shape
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            logger.warning("invalid_bbox", bbox=bbox, image_shape=(h, w))
            return None

        # extract bbox center
        bbox_w, bbox_h = x2 - x1, y2 - y1
        roi_w = max(10, int(bbox_w * self.config.roi_percentage))
        roi_h = max(10, int(bbox_h * self.config.roi_percentage))

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        roi_x1 = max(0, center_x - roi_w // 2)
        roi_y1 = max(0, center_y - roi_h // 2)
        roi_x2 = min(w, center_x + roi_w // 2)
        roi_y2 = min(h, center_y + roi_h // 2)

        # extract ROI
        roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0:
            return None

        # convert depth values to meters
        roi_meters = roi.astype(np.float32) / 1000.00

        # filter valid
        valid_mask = (
                (roi_meters >= self.config.depth_range.min) &
                (roi_meters <= self.config.depth_range.max) &
                (roi_meters > 0)
        )

        valid_depths = roi_meters[valid_mask]

        if len(valid_depths) == 0:
            logger.debug("no_valid_depth_values", roi_shape=roi.shape)
            return None

        # check if we have sufficient valid pixels
        valid_ratio = len(valid_depths) / roi.size
        if valid_ratio < self.config.min_valid_pixels_ratio:
            logger.debug("insufficient_valid_depth_pixels", valid_ratio=valid_ratio, required=self.config.min_valid_pixels_ratio)
            return None

        # outlier rejection using statistical filtering
        depth_stats = self._calculate_depth_statistics(valid_depths)
        filtered_depths = self._reject_outliers(valid_depths, depth_stats)

        if len(filtered_depths) < max(3, len(valid_depths) * 0.3):
            logger.debug("too_many_outliers_rejected")
            return None

        # final depth value (median for robustness)
        final_depth = float(np.median(filtered_depths))

        # calculate confidence based data quality
        confidence = self._calculate_depth_confidence(
            filtered_depths, depth_stats, valid_ratio
        )

        roi_stats = {
            'roi_size': roi.size,
            'valid_pixels': len(valid_depths),
            'valid_ratio': valid_ratio,
            'depth_std': depth_stats['std'],
            'outliers_removed': len(valid_depths) - len(filtered_depths)
        }

        return final_depth, confidence, roi_stats

    def _calculate_depth_statistics(self, depths: np.ndarray) -> Dict[str, float]:
        """calculate statistic for depth values"""
        return {
            'mean': float(np.mean(depths)),
            'median': float(np.median(depths)),
            'std': float(np.std(depths)),
            'min': float(np.min(depths)),
            'max': float(np.max(depths)),
            'count': len(depths)
        }

    def _reject_outliers(
            self,
            depths: np.ndarray,
            stats: Dict[str, float]
    ) -> np.ndarray:
        """Remove outliers from depth values"""

        mean_depth = stats['mean']
        std_depth = stats['std']

        # use a configurable sigma threshold for outlier rejection
        threshold = self.config.outlier_rejection_sigma * std_depth

        # keep values within threshold
        mask = np.abs(depths - mean_depth) < threshold
        return depths[mask]

    def _calculate_depth_confidence(
            self,
            filtered_depths: np.ndarray,
            stats: Dict[str, float],
            valid_ratio: float
    ) -> float:
        """calculate confidence level for depth values"""
        confidence = valid_ratio

        # penalty for high variance (low precision)
        if stats['std'] > 0:
            std_penalty = min(0.3, stats['std'] / stats['mean'])
            confidence *= (1.0 - std_penalty)

        # bonus for sufficient data points
        if len(filtered_depths) > 50:
            confidence = min(1.0, confidence * 1.1)

        # Penalty for very close or very far subjects
        mean_depth = stats['mean']
        if mean_depth < 1.5 or mean_depth > 6.0:
            confidence *= 0.8

        return float(np.clip(confidence, 0.0, 1.0))

    def _get_roi_from_bbox(
            self,
            bbox: Tuple[int, int, int, int],
            depth_image: np.ndarray,
            size_factor: float = 0.4
    ) -> Optional[np.ndarray]:
        """Extract roi from bbox with given size factor"""
        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1

        # cal ROI bounds
        roi_width = int(width * size_factor)
        roi_height = int(height * size_factor)

        roi_x1 = max(0, center_x - roi_width // 2)
        roi_y1 = max(0, center_y - roi_height // 2)
        roi_x2 = min(depth_image.shape[1], center_x + roi_width // 2)
        roi_y2 = min(depth_image.shape[0], center_y + roi_height // 2)

        roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0:
            return None

        return roi

    # Extraction levels:
    #         1. Center ROI (40% of bbox) - Most reliable
    #         2. Expanded ROI (60% of bbox) - Fallback if center fails
    #         3. Statistical filtering - Aggressive filtering on full bbox
    #         4. Bbox-based estimate - Last resort

    async def _extract_center_roi_depth(
            self,
            bbox: Tuple[int, int, int, int],
            depth_image: np.ndarray,
    ) -> Optional[Tuple[float, float, Dict[str, Any]]]:
        """Extract depth from center 40% of bbox """

        roi = self._get_roi_from_bbox(bbox, depth_image, size_factor=0.4)
        if roi is None:
            return None

        # check quality
        is_valid, quality_score, reason = self.quality_checker.check_depth_quality(roi)
        if not is_valid:
            logger.debug("center_roi_quality_check_failed", reason=reason)
            return None

        # extract valid depths
        min_depth = self.config.depth_range.min * 1000
        max_depth = self.config.depth_range.max * 1000
        valid_depths = roi[(roi > min_depth) & (roi < max_depth)]

        if len(valid_depths) == 0:
            return None

        # statistical filtering
        depth_stats = self._calculate_depth_statistics(valid_depths)
        filtered_depths = self._reject_outliers(valid_depths, depth_stats)

        if len(filtered_depths) < 3:
            return None

        final_depth = float(np.median(filtered_depths)) / 1000.0  # to meters

        # cal conf
        valid_ratio = len(valid_depths) / roi.size
        confidence = self._calculate_depth_confidence(
            filtered_depths, depth_stats, valid_ratio
        )
        confidence *= quality_score  # factor in quality score

        roi_stats = {
            'roi_size': roi.size,
            'valid_pixels': len(valid_depths),
            'valid_ratio': valid_ratio,
            'quality_score': quality_score,
            'depth_std': depth_stats['std'] / 1000.0,
            'outliers_removed': len(valid_depths) - len(filtered_depths)
        }

        return final_depth, confidence, roi_stats

    async def _extract_expanded_roi_depth(
            self,
            bbox: Tuple[int, int, int, int],
            depth_image: np.ndarray,
    ) -> Optional[Tuple[float, float, Dict[str, Any]]]:
        """Extract depth from expanded 60% ROI (fallback method)."""
        roi = self._get_roi_from_bbox(bbox, depth_image, size_factor=0.6)
        if roi is None:
            return None

        # Check quality
        is_valid, quality_score, reason = self.quality_checker.check_depth_quality(roi)
        if not is_valid:
            return None

        min_depth = self.config.depth_range.min * 1000
        max_depth = self.config.depth_range.max * 1000
        valid_depths = roi[(roi > min_depth) & (roi < max_depth)]

        if len(valid_depths) < 10:  # Need more pixels for expanded ROI
            return None

        depth_stats = self._calculate_depth_statistics(valid_depths)
        filtered_depths = self._reject_outliers(valid_depths, depth_stats)

        if len(filtered_depths) < 5:
            return None

        final_depth = float(np.median(filtered_depths)) / 1000.0

        valid_ratio = len(valid_depths) / roi.size
        confidence = self._calculate_depth_confidence(
            filtered_depths, depth_stats, valid_ratio
        )
        confidence *= quality_score * 0.85  # Penalty for using fallback method

        roi_stats = {
            'roi_size': roi.size,
            'valid_pixels': len(valid_depths),
            'valid_ratio': valid_ratio,
            'quality_score': quality_score,
            'depth_std': depth_stats['std'] / 1000.0,
            'outliers_removed': len(valid_depths) - len(filtered_depths)
        }

        return final_depth, confidence, roi_stats

    async def _extract_statistical_depth(
            self,
            bbox: Tuple[int, int, int, int],
            depth_image: np.ndarray
    ) -> Optional[Tuple[float, float, Dict[str, Any]]]:
        """Extract depth using aggressive statistical filtering (uses full bbox)."""

        # Use full bbox
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(depth_image.shape[1], x2)
        y2 = min(depth_image.shape[0], y2)

        roi = depth_image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Get valid depths
        min_depth = self.config.depth_range.min * 1000
        max_depth = self.config.depth_range.max * 1000
        valid_depths = roi[(roi > min_depth) & (roi < max_depth)]

        if len(valid_depths) < 20:  # Need significant data
            return None

        # More aggressive outlier removal
        depth_stats = self._calculate_depth_statistics(valid_depths)

        # Use stricter sigma
        mean_depth = depth_stats['mean']
        std_depth = depth_stats['std']
        threshold = 1.5 * std_depth  # Stricter than normal 2.0

        mask = np.abs(valid_depths - mean_depth) < threshold
        filtered_depths = valid_depths[mask]

        if len(filtered_depths) < 10:
            return None

        final_depth = float(np.median(filtered_depths)) / 1000.0

        # Lower confidence due to aggressive filtering
        valid_ratio = len(valid_depths) / roi.size
        confidence = min(0.7, valid_ratio * 0.8)

        roi_stats = {
            'roi_size': roi.size,
            'valid_pixels': len(valid_depths),
            'valid_ratio': valid_ratio,
            'quality_score': 0.6,
            'depth_std': depth_stats['std'] / 1000.0,
            'outliers_removed': len(valid_depths) - len(filtered_depths)
        }

        return final_depth, confidence, roi_stats

    def _estimate_depth_from_bbox(
            self,
            bbox: Tuple[int, int, int, int],
            depth_image: np.ndarray,
    ) -> Optional[Tuple[float, float, Dict[str, Any]]]:
        """Estimate depth from bbox --> by assuming human height ~= 1.7m"""

        x1, y1, x2, y2 = bbox
        bbox_height_pixels = y2 - y1

        if bbox_height_pixels < 50:
            return None

        if not self.intrinsics_received:
            return None

        # estimate: person height / bbox height ≈ distance
        estimated_distance = (1.7 * self.intrinsics.fy) / bbox_height_pixels

        confidence = 0.3  # low since estimate

        roi_stats = {
            'roi_size': 0,
            'valid_pixels': 0,
            'valid_ratio': 0.0,
            'quality_score': 0.3,
            'depth_std': 0.0,
            'outliers_removed': 0,
            'bbox_height_pixels': bbox_height_pixels,
            'estimation_method': 'bbox_size'
        }

        return estimated_distance, confidence, roi_stats

    async def _extract_depth_multi_level(
            self,
            bbox: Tuple[int, int, int, int],
            depth_image: np.ndarray,
    ) -> Optional[Tuple[float, float, Dict[str, Any], str]]:
        """multi-level depth extraction with fallbacks"""

        # Extraction levels:
        #         1. Center ROI (40% of bbox) - Most reliable
        #         2. Expanded ROI (60% of bbox) - Fallback if center fails
        #         3. Statistical filtering - Aggressive filtering on full bbox
        #         4. Bbox-based estimate - Last resort

        # Lvl 1: Center ROI --> most reliable
        result = await self._extract_center_roi_depth(bbox, depth_image)
        if result is not None:
            depth, confidence, stats = result
            stats['extraction_level'] = 1
            self.processing_stats['extraction_method_counts']['center_roi'] += 1
            return depth, confidence, stats, "center_roi"

        # Lvl 2: Expanded ROI
        result = await self._extract_expanded_roi_depth(bbox, depth_image)
        if result is not None:
            depth, confidence, stats = result
            stats['extraction_level'] = 2
            self.processing_stats['extraction_method_counts']['expanded_roi'] += 1
            return depth, confidence, stats, "expanded_roi"

        # Lvl 3: Statistical filtering with full bbox
        result = await self._extract_statistical_depth(bbox, depth_image)
        if result is not None:
            depth, confidence, stats = result
            stats['extraction_level'] = 3
            self.processing_stats['extraction_method_counts']['statistical'] += 1
            return depth, confidence, stats, "statistical"

        # Lvl 4: Estimate if everything fails
        result = self._estimate_depth_from_bbox(bbox, depth_image)
        if result is not None:
            depth, confidence, stats = result
            stats['extraction_level'] = 4
            self.processing_stats['extraction_method_counts']['fallback_estimate'] += 1
            return depth, confidence, stats, "fallback_estimate"

        # if all levels failed
        self.processing_stats['failed_extractions'] += 1
        return None

    async def extract_multiple_depths(
            self,
            bboxes: List[Tuple[int, int, int, int]],
            depth_image: Optional[np.ndarray] = None
    ) -> List[Optional[DepthData]]:
        """Extract depth values from multiple bboxes concurrently"""

        if not bboxes:
            return []

        tasks = [
            self.extract_person_depth(bbox, depth_image) for bbox in bboxes
        ]

        # extract all extractions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # process results, handling exceptions
        depth_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "depth_extraction_exception",
                    bbox_index=i,
                    bbox=bboxes[i],
                    error=str(result)
                )
                depth_results.append(None)
            else:
                depth_results.append(result)

        #log batch statistics
        successful = sum(1 for r in depth_results if r is not None)
        logger.debug(
            "batch_depth_extraction_complete",
            total=len(bboxes),
            successful=successful,
            failed=len(bboxes) - successful
        )

        return depth_results

    def _update_performance_metrics(self, processing_time_ms: float) -> None:
        """Update performance tracking metrics."""
        self.processing_count += 1
        self.total_processing_time += processing_time_ms

        # Check performance thresholds
        target_time = self.settings.testing.performance_benchmarks.depth_processing_ms
        if processing_time_ms > target_time:
            logger.warning(
                "slow_depth_processing",
                processing_time_ms=processing_time_ms,
                target_ms=target_time
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Return performance statistics."""
        if self.processing_count == 0:
            return {
                'processing_count': 0,
                'frames_processed': self.processing_stats['frames_processed']
            }

        avg_time = self.total_processing_time / self.processing_count

        #cal success rate
        failed = self.processing_stats['failed_extractions']
        success_rate = ((self.processing_count - failed) / self.processing_count * 100) if self.processing_count > 0 else 0

        return {
            'processing_count': self.processing_count,
            'average_processing_time_ms': round(avg_time, 2),
            'total_processing_time_ms': round(self.total_processing_time, 2),
            'target_time_ms': self.settings.testing.performance_benchmarks.depth_processing_ms,
            'intrinsics_available': self.intrinsics_received,
            'latest_depth_age_ms': round((time.time() - self.depth_timestamp) * 1000, 1) if self.depth_timestamp > 0 else None,
            'frames_processed': self.processing_stats['frames_processed'],
            'failed_extractions': failed,
            'success_rate_percent': round(success_rate, 1),
            'extraction_methods': self.processing_stats['extraction_method_counts']
        }

    def set_depth_image(self, depth_image: np.ndarray) -> None:
        """Set depth image. (Manual test without ROS)"""
        self.latest_depth_image = depth_image
        self.depth_timestamp = time.time()

    def set_camera_intrinsics(
            self,
            fx: float,
            fy: float,
            cx: float,
            cy: float,
            width: int,
            height: int,
    ) -> None:
        """Manually  set the camera intrinsics. (Manual test without ROS)"""

        class MockCameraInfo:
            def __init__(self):
                self.k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                self.width = width
                self.height = height

        self.intrinsics = CameraIntrinsics(MockCameraInfo())
        self.intrinsics_received = True

        logger.info("camera_intrinsics_set_manually",
                    fx=fx, fy=fy, cx=cx, cy=cy, resolution=(width, height))

    @log_performance("depth_validation")
    async def validate_depth_processing(self) -> Dict[str, Any]:
        """Validate depth processing with test data."""
        if not self.intrinsics_received:
            return {
                'validation_successful': False,
                'error': 'Camera intrinsics not available'
            }

        if self.latest_depth_image is None:
            return {
                'validation_successful': False,
                'error': 'No depth image available'
            }

            # Test with sample bounding box
        h, w = self.latest_depth_image.shape
        test_bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)  # Center quarter of image

        try:
            depth_data = await self.extract_person_depth(test_bbox)

            validation_result = {
                'validation_successful': depth_data is not None,
                'depth_image_shape': self.latest_depth_image.shape,
                'test_bbox': test_bbox,
                'performance_stats': self.get_performance_stats()
            }

            if depth_data:
                validation_result.update({
                    'test_depth_value': depth_data.depth_value,
                    'test_confidence': depth_data.confidence,
                    'test_position_3d': depth_data.position_3d,
                    'processing_time_ms': depth_data.processing_time_ms
                })

            logger.info("depth_processing_validation_completed", **validation_result)
            return validation_result

        except Exception as e:
            logger.error("depth_validation_failed", error=str(e))
            return {
                'validation_successful': False,
                'error': str(e)
            }


# Convenience function
async def extract_person_depth(
        bbox: Tuple[int, int, int, int],
        depth_image: np.ndarray,
        camera_intrinsics: Optional[CameraIntrinsics] = None
) -> Optional[DepthData]:
    """
    Convenience function for depth extraction.
    (temporary processor instance.)
    """
    processor = DepthProcessor()

    if camera_intrinsics:
        processor.intrinsics = camera_intrinsics
        processor.intrinsics_received = True

    processor.set_depth_image(depth_image)

    return await processor.extract_person_depth(bbox)

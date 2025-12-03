import asyncio
import time
from asyncio import set_child_watcher

import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
import cv2
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from ..utils.logger import get_logger, log_performance
from ..utils.threading_utils import get_thread_manager, async_retry
from ..utils.exceptions import (
    ModelLoadError,
    ModelInferenceError,
    DetectionError,
    PerformanceError
)

logger = get_logger(__name__)

class PersonDetection:
    """Data class for person detection."""

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        pose_keypoints: np.ndarray,
        pose_confidence: np.ndarray,
        track_id: Optional[int] = None,
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.pose_keypoints = pose_keypoints
        self.pose_confidence = pose_confidence
        self.track_id = track_id

        #calculate derived properties
        self.center = self._calculate_center()
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.area = self.width * self.height
        self.aspect_ratio = self.width / self.height if self.height > 0 else 0

        #pose analytics
        self.pose_analysis = self._analyze_pose()


    def _calculate_center(self) -> Tuple[float, float]:
        """Calculate center of person detection."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3] / 2)
        )

    def _analyze_pose(self) -> Dict[str, Union[float, bool, int]]:
        """Analyze person detection result."""

        # COCO keypoint indices
        LEFT_SHOULDER = 5,
        RIGHT_SHOULDER = 6,
        NOSE = 0,
        LEFT_EYE = 1,
        RIGHT_EYE = 2,

        analysis = {
            'visible_keypoints': int(np.sum(self.pose_confidence > 0.5)),
            'pose_quality': float(np.mean(self.pose_confidence)),
            'body_facing_camera': None,
            'shoulder_angle': None,
            'upper_body_visible': False
        }

        #check upper body visibility
        if (self.pose_confidence[LEFT_SHOULDER] > 0.5 and self.pose_confidence[RIGHT_SHOULDER] > 0.5):
            analysis['upper_body_visible'] = True

            #calculate shoulder angle
            left_shoulder = self.pose_keypoints[LEFT_SHOULDER]
            right_shoulder = self.pose_keypoints[RIGHT_SHOULDER]

            shoulder_vector = right_shoulder - left_shoulder
            shoulder_angle = float(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
            analysis['shoulder_angle'] = shoulder_angle

            #determine if body is facing the camera (shoulder angle should close to horizontal)
            analysis['body_facing_camera'] = abs(shoulder_angle) < 0.3 # -17 degrees

        return analysis

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'pose_keypoints': self.pose_keypoints.tolist(),
            'pose_confidence': self.pose_confidence.tolist(),
            'track_id': self.track_id,
            'center': self.center,
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'aspect_ratio': self.aspect_ratio,
            'pose_analysis': self.pose_analysis
        }

class YOLOPoseDetector:

    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[YOLO] = None
        self.model_loaded = False
        self.thread_manager = get_thread_manager()

        #performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_inference_time = 0.0

        #model configuration
        self.config = self.settings.vision.yolo

        logger.info("yolo_detector_initialized", config=self.config.model_dump())

    async def load_model(self) -> None:
        """Load model from disk."""
        if self.model_loaded:
            logger.warning("Model already loaded", model=self.model)
            return

        if not ULTRALYTICS_AVAILABLE:
            raise ModelLoadError("Ultralytics is not available")

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise ModelInferenceError(f"Model file not found: {model_path}")

        try:
            logger.info(f"loading_yolo_model", path=str(model_path))

            #load model in thread pool
            def _load_model():
                model = YOLO(str(model_path), task='pose')

                #warm up
                dummy_input = np.zeros((640, 640, 3), dtype = np.uint8)
                model(dummy_input, verbose=False)
                return model

            self.model = await self.thread_manager.run_vision_task(_load_model)
            self.model_loaded = True

            logger.info(f"yolo_model_loaded", model=self.model)

        except Exception as e:
            logger.error(f"yolo_model_loading_failed: {e}", error=str(e))
            raise ModelLoadError(f"Failed to load YOLO model: {e}") from e


    @async_retry(max_attempts=3, delay=0.1, exceptions = (ModelInferenceError,))
    async def detect_async(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[PersonDetection]:
        """Detect person using YOLO model with pose estimation."""

        if not self.model_loaded:
            raise ModelInferenceError("Model not loaded. Load the model first.")

        if frame is None or frame.size == 0:
            raise ModelInferenceError("Invalid input frame.")

        confidence_threshold = confidence_threshold or self.config.confidence_threshold

        start_time = time.perf_counter()

        try:
            results = await self.thread_manager.run_vision_task(
                self._sync_inference, frame, confidence_threshold
            )

            inference_time = time.perf_counter() - start_time
            self._update_performance_metrices(inference_time)

            detections = self._process_results(results)

            logger.debug(
                "detection_completed",
                detections_count=len(detections),
                inference_time_ms=round(inference_time * 1000, 2),
                confidence_threshold=confidence_threshold
            )

            return detections

        except Exception as e:
            logger.error("inference_failed", error=str(e), frame_shape=frame.shape)
            raise ModelInferenceError(f"YOLO inference failed: {e}") from e

    def _sync_inference(self, frame: np.ndarray, confidence_threshold: float):
        """Synchronize inference results with YOLO model for thread pool."""
        return self.model(
            frame,
            conf=confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            verbose=False,
            imgsz=self.config.input_size,
        )

    def _process_results(self, results) -> List[PersonDetection]:
        """Process results from YOLO model."""
        detections = []

        if not results or len(results) == 0:
            return detections

        result = results[0]

        boxes = result.boxes
        keypoints = result.keypoints

        #filter for person class
        person_mask = boxes.cls == 0

        if not person_mask.any():
            return detections

        person_boxes = boxes[person_mask]
        person_keypoints = keypoints[person_mask]

        for i in range(len(person_boxes)):
            box = person_boxes[i]
            kpts = person_keypoints[i]

            #extrach bbox
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])

            #extract keypoints
            keypoints_xy = kpts.xy[0].cpu().numpy()
            keypoints_conf = kpts.conf[0].cpu().numpy()

            #
            detection = PersonDetection(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                pose_keypoints=keypoints_xy,
                pose_confidence=keypoints_conf
            )

            detections.append(detection)

        return detections

    def _update_performance_metrices(self, inference_time: float) -> None:
        """Update performance metrics."""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.last_inference_time = inference_time

        # check performance thresholds
        if inference_time > 0.050:  # 50ms threshold
            logger.warning(
                "slow_inference",
                inference_time_ms=round(inference_time * 1000, 2),
                threshold_ms=50
            )


    @log_performance("model_validation")
    async def validate_model(self) -> Dict[str, Union[bool, float, str]]:
        """Validate the YOLO model."""
        if not self.model_loaded:
            await self.load_model()

        #create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        try:
            start_time = time.perf_counter()
            detections = await self.detect_async(test_image)
            inference_time = time.perf_counter() - start_time

            validation_result = {
                'model_loaded': True,
                'inference_successful': True,
                'inference_time_ms': round(inference_time * 1000, 2),
                'detections_count': len(detections),
                'meets_performance_target': inference_time < 0.020,  # 20ms target
                'model_path': str(self.config.model_path)
            }

            logger.info("model_validation_completed", **validation_result)
            return validation_result

        except Exception as e:
            logger.error("model_validation_failed", error=str(e))
            return {
                'model_loaded': self.model_loaded,
                'inference_successful': False,
                'error': str(e)
            }

    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """Get performance statistics."""
        if self.inference_count == 0:
            return {'inference_count': 0}

        avg_time = self.total_inference_time / self.inference_count
        fps_capability = 1.0 / avg_time if avg_time > 0 else 0

        return {
            'inference_count': self.inference_count,
            'total_inference_time': round(self.total_inference_time, 3),
            'average_inference_time_ms': round(avg_time * 1000, 2),
            'last_inference_time_ms': round(self.last_inference_time * 1000, 2),
            'fps_capability': round(fps_capability, 1),
            'meets_target_fps': fps_capability >= self.settings.performance.target_fps
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("cleaning_up_yolo_detector")

        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False

        logger.info(f"yolo_detector_cleanup_completed")


# Convenience function for quick detection

async def detect_persons_with_pose(frame: np.ndarray) -> List[PersonDetection]:
    """Detect person using YOLO model with pose estimation. -temporary"""

    detector = YOLOPoseDetector()
    await detector.load_model()

    try:
        return await detector.detect_async(frame)
    finally:
        await detector.cleanup()


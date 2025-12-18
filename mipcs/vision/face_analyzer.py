import asyncio
import time
from pathlib import Path
import sys
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
import cv2


try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from ..utils.logger import get_logger
from ..utils.threading_utils import get_thread_manager, async_retry
from ..utils.exceptions import (
    FaceAnalysisError,
    ModelLoadError,
    PerformanceError
)

logger = get_logger(__name__)

class FaceAnalysis:
    """Dataclass"""

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        facing_camera: bool,
        orientation: str,
        landmarks: Optional[np.ndarray] = None,
        confidence: float = 0.0,
        analysis_metrics: Optional[Dict] = None,
    ):
        self.bbox = bbox
        self.facing_camera = facing_camera
        self.orientation = orientation  # 'frontal', 'profile', 'partial_profile', 'unknown'
        self.landmarks = landmarks
        self.confidence = confidence
        self.analysis_metrics = analysis_metrics or {}

        #calculated properties
        self.face_center = self._calculate_center()
        self.face_width = bbox[2] - bbox[0]
        self.face_height = bbox[3] - bbox[1]

    def _calculate_center(self) -> Tuple[float, float]:
        """Calculate center of person detection."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3] / 2)
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'bbox': self.bbox,
            'facing_camera': self.facing_camera,
            'orientation': self.orientation,
            'confidence': self.confidence,
            'face_center': self.face_center,
            'face_width': self.face_width,
            'face_height': self.face_height,
            'analysis_metrics': self.analysis_metrics,
            'landmarks_detected': self.landmarks is not None
        }

class MediaPipeFaceAnalyzer:

    def __init__(self):
        self.settings = get_settings()
        self.config = self.settings.vision.mediapipe
        self.thread_manager = get_thread_manager()

        #mediapipe components
        self.face_mesh = None
        self.face_detection = None
        self.mp_face_mesh = None
        self.mp_face_detection = None
        self.mp_drawing = None

        self.models_loaded = False

        #performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0

        logger.info("mediapipe_face_analyzer_initialized")

    async def load_models(self) -> None:
        """Load mediapipe models."""
        if self.models_loaded:
            logger.warning("models_already_loaded")
            return

        if not MEDIAPIPE_AVAILABLE:
            raise ModelLoadError("mediapipe library is not available")

        try:
            logger.info("loading_mediapipe_models")

            def _load_models():
                mp_face_mesh = mp.solutions.face_mesh
                mp_face_detection = mp.solutions.face_detection
                mp_drawing = mp.solutions.drawing_utils

                face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=self.config.face_mesh.max_num_faces,
                    refine_landmarks=self.config.face_mesh.refine_landmarks,
                    min_detection_confidence=self.config.face_mesh.min_detection_confidence,
                    min_tracking_confidence=self.config.face_mesh.min_tracking_confidence,
                    static_image_mode=self.config.face_mesh.static_image_mode
                )

                face_detection = mp_face_detection.FaceDetection(
                    model_selection=self.config.face_detection.model_selection,
                    min_detection_confidence=self.config.face_detection.min_detection_confidence,
                )

                return {
                    'face_mesh': face_mesh,
                    'face_detection': face_detection,
                    'mp_face_mesh': mp_face_mesh,
                    'mp_face_detection': mp_face_detection,
                    'mp_drawing': mp_drawing
                }

            components = await self.thread_manager.run_vision_task(_load_models)

            self.face_mesh = components['face_mesh']
            self.face_detection = components['face_detection']
            self.mp_face_mesh = components['mp_face_mesh']
            self.mp_face_detection = components['mp_face_detection']
            self.mp_drawing = components['mp_drawing']

            self.models_loaded = True
            logger.info("mediapipe_models_loaded_successfully")

        except Exception as e:
            logger.error("mediapipe_models_loading_failed", error=str(e))
            raise ModelLoadError(f"Failed to load MediaPipe models: {e}") from e

    @async_retry(max_attempts=2, delay=0.1, exceptions=(FaceAnalysisError, ))
    async def analyze_face_orientation(
            self,
            frame: np.ndarray,
            person_bbox: Tuple[int, int, int, int],
        ) -> Optional[FaceAnalysis]:
        """Analyze face orientation within person bbox"""

        if not self.models_loaded:
            raise FaceAnalysisError("mediapipe_model_not_loaded")

        if frame is None or frame.size == 0:
            raise FaceAnalysisError("Invalid input frame")

        start_time = time.perf_counter()

        try:
            face_region, region_bbox = self._extract_face_region(frame, person_bbox)

            if face_region is None:
                logger.debug("no_valid_face_region", person_bbox = person_bbox)
                return None

            #analysis face in thread pool
            analysis = await self.thread_manager.run_vision_task(
                self._sync_face_analysis, face_region, region_bbox
            )

            analysis_time = time.perf_counter() - start_time
            self._update_performance_metrices(analysis_time)

            if analysis:
                logger.debug(
                    "face_analysis_completed",
                    facing_camera=analysis.facing_camera,
                    orientation=analysis.orientation,
                    confidence=analysis.confidence,
                    analysis_time_ms=round(analysis_time * 1000, 2)
                )

            return analysis

        except Exception as e:
            logger.error("face_analysis_failed", error=str(e), person_bbox=person_bbox)
            raise FaceAnalysisError(f"Face analysis failed: {e}") from e

    def _extract_face_region(
            self,
            frame: np.ndarray,
            person_bbox: Tuple[int, int, int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Extract face region with padding around person's upper body"""

        x1, y1, x2, y2 = person_bbox
        h, w = frame.shape[:2]

        #expand region for face detection (upper portion)
        person_height = y2 - y1
        face_region_height = min(int(person_height * 0.4), 200) # upper 40% or max 200px

        #calculate face region bounds
        face_x1 = max(0, x1 - 20)  # Small horizontal padding
        face_y1 = max(0, y1 - 30)  # Extended upward for head
        face_x2 = min(w, x2 + 20)
        face_y2 = min(h, y1 + face_region_height)

        if face_x2 <= face_x1 or face_y2 <= face_y1:
            return None, None

        face_region = frame[face_y1:face_y2, face_x1:face_x2].copy()
        region_bbox = (face_x1, face_y1, face_x2, face_y2)

        return face_region, region_bbox

    def _sync_face_analysis(
            self,
            face_region: np.ndarray,
            region_bbox: Tuple[int, int, int, int]
    ) -> Optional[FaceAnalysis]:
        """Synchronize face region and region_bbox with face analysis"""

        #convert BGR to RGB for mediapipe
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

        #try face mesh first (detailed)
        mesh_results = self.face_mesh.process(face_rgb)

        if mesh_results.multi_face_landmarks:
            return self._analyze_with_face_mesh(
                mesh_results, face_region, region_bbox
            )

        #fallback to face detection
        detection_results = self.face_detection.process(face_rgb)

        if detection_results.detections:
            return self._analyze_with_face_detection(
                detection_results, face_region, region_bbox
            )

        return None

    def _analyze_with_face_mesh(
            self,
            mesh_results,
            face_region: np.ndarray,
            region_bbox: Tuple[int, int, int, int]
    ) -> FaceAnalysis:
        """Analyze face orientation with face mesh landmarks"""

        landmarks = mesh_results.multi_face_landmarks[0]
        h, w = face_region.shape[:2]

        #key landmarks for orientation analyze
        NOSE_TIP = 1
        LEFT_EYE_CORNER = 33
        RIGHT_EYE_CORNER = 263
        CHIN = 175
        LEFT_CHEEK = 116
        RIGHT_CHEEK = 345

        #copnvert normalized landmarks to pixel coords
        nose_tip = np.array([
            landmarks.landmark[NOSE_TIP].x * w,
            landmarks.landmark[NOSE_TIP].y * h,
        ])
        left_eye = np.array([
            landmarks.landmark[LEFT_EYE_CORNER].x * w,
            landmarks.landmark[LEFT_EYE_CORNER].y * h,
        ])
        right_eye = np.array([
            landmarks.landmark[RIGHT_EYE_CORNER].x * w,
            landmarks.landmark[RIGHT_EYE_CORNER].y * h,
        ])

        #calculate orientation metrics
        eye_distance = np.linalg.norm(right_eye - left_eye)
        eye_center = (left_eye + right_eye) / 2
        nose_offset = nose_tip[0] - eye_center[0]

        #face width estimation (normalized to region width)
        face_width_ratio = eye_distance / w

        #determine orientation and camera facing status
        orientation, facing_camera, confidence = self._classify_orientation(
            face_width_ratio, nose_offset, eye_distance
        )

        #calculate face bbox within range
        face_bbox = self._calculate_face_bbox_from_landmarks(landmarks, face_region.shape[:2])

        #adjust bbox to full frame coords
        adjusted_bbox = (
            region_bbox[0] + face_bbox[0],
            region_bbox[1] + face_bbox[1],
            region_bbox[0] + face_bbox[2],
            region_bbox[1] + face_bbox[3]
        )

        #prepare landmarks array
        landmarks_array = np.array([
            [lm.x * w, lm.y *h, lm.z] for lm in landmarks.landmark
        ])

        analysis_metrics = {
            'eye_distance': float(eye_distance),
            'nose_offset': float(nose_offset),
            'face_width_ratio': float(face_width_ratio),
            'landmark_count': len(landmarks.landmark),
            'method': 'face_mesh'
        }

        return FaceAnalysis(
            bbox=adjusted_bbox,
            facing_camera=facing_camera,
            orientation=orientation,
            landmarks=landmarks_array,
            confidence=confidence,
            analysis_metrics=analysis_metrics,
        )

    def _analyze_with_face_detection(
            self,
            detection_results,
            face_region: np.ndarray,
            region_bbox: Tuple[int, int, int, int]
    ) -> FaceAnalysis:
        """Analyze face orientation using basic face detection."""
        detection = detection_results.detections[0]
        h, w = face_region.shape[:2]

        # Extract face bounding box
        bbox = detection.location_data.relative_bounding_box
        face_x1 = int(bbox.xmin * w)
        face_y1 = int(bbox.ymin * h)
        face_w = int(bbox.width * w)
        face_h = int(bbox.height * h)

        face_x2 = face_x1 + face_w
        face_y2 = face_y1 + face_h

        # Adjust to full frame coordinates
        adjusted_bbox = (
            region_bbox[0] + face_x1,
            region_bbox[1] + face_y1,
            region_bbox[0] + face_x2,
            region_bbox[1] + face_y2
        )

        # Basic orientation classification based on face aspect ratio
        face_aspect_ratio = face_w / face_h if face_h > 0 else 0
        face_width_ratio = face_w / w

        orientation, facing_camera, confidence = self._classify_orientation_basic(
            face_aspect_ratio, face_width_ratio
        )

        analysis_metrics = {
            'face_aspect_ratio': float(face_aspect_ratio),
            'face_width_ratio': float(face_width_ratio),
            'detection_confidence': float(detection.score[0]),
            'method': 'face_detection'
        }

        return FaceAnalysis(
            bbox=adjusted_bbox,
            facing_camera=facing_camera,
            orientation=orientation,
            landmarks=None,
            confidence=confidence,
            analysis_metrics=analysis_metrics
        )

    def _classify_orientation(
            self,
            face_width_ratio: float,
            nose_offset: float,
            eye_distance: float
    ) -> Tuple[str, bool, float]:
        """Classify face orientation based  on facial landmarks."""
        FRONTAL_THRESHOLD = 0.08
        PARTIAL_THRESHOLD = 0.04
        NOSE_CENTER_THRESHOLD = 10

        confidence = 0.7

        if face_width_ratio > FRONTAL_THRESHOLD:
            #frontal face detection
            if abs(nose_offset) < NOSE_CENTER_THRESHOLD:
                orientation = 'frontal'
                facing_camera = True
                confidence = 0.9
            elif nose_offset > NOSE_CENTER_THRESHOLD:
                orientation = "slight_left_turn"
                facing_camera = True
                confidence = 0.8
            else:
                orientation = "slight_right_turn"
                facing_camera = True
                confidence = 0.8

        elif face_width_ratio > PARTIAL_THRESHOLD:
            #partial profile
            orientation = "partial_profile"
            facing_camera = False
            confidence = 0.7

        else:
            # Full profile or very poor detection
            orientation = "profile"
            facing_camera = False
            confidence = 0.6

        return orientation, facing_camera, confidence

    def _classify_orientation_basic(
            self,
            face_aspect_ratio: float,
            face_width_ratio: float
    ) -> Tuple[str, bool, float]:
        """Classify face orientation without facial landmarks"""
        # Simple heuristics based on face detection only
        if face_width_ratio > 0.15 and 0.8 < face_aspect_ratio < 1.2:
            # Likely frontal face
            orientation = "frontal"
            facing_camera = True
            confidence = 0.7
        elif face_width_ratio > 0.08:
            # Partial profile
            orientation = "partial_profile"
            facing_camera = True
            confidence = 0.6
        else:
            # Profile or poor detection
            orientation = "profile"
            facing_camera = False
            confidence = 0.5

        return orientation, facing_camera, confidence

    def _calculate_face_bbox_from_landmarks(
            self,
            landmarks,
            image_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Calculate face bbox based on landmarks."""

        h, w = image_shape

        #extract x, y coords of all landmarks
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]

        #calculate bbox with small padding
        padding = 5
        x1 = max(0, int(min(x_coords)) - padding)
        y1 = max(0, int(min(y_coords)) - padding)
        x2 = min(w, int(max(x_coords)) + padding)
        y2 = min(h, int(max(y_coords)) + padding)

        return (x1, y1, x2, y2)

    def _update_performance_metrices(
            self,
            analysis_time: float,
    ) -> None:
        """Update performance metrices."""
        self.analysis_count += 1
        self.total_analysis_time += analysis_time

        #check performance thresholds
        target_time = self.settings.testing.performance_benchmarks.face_analysis_ms / 1000.0
        if analysis_time > target_time:
            logger.warning(
                "slow_face_analysis",
                analysis_time_ms=round(analysis_time * 1000, 2),
                target_ms=target_time * 1000
            )

    async def analyze_multiple_faces(
            self,
            frame: np.ndarray,
            person_bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Optional[FaceAnalysis]]:
        """Analyze multiple face detection results."""
        if not person_bboxes:
            return []

        #create concurrent analysis tasks
        tasks = [
            self.analyze_face_orientation(frame, bbox) for bbox in person_bboxes
        ]

        #Excute all analysis concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        #Process results, handling exceptions
        face_analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "face_analysis_exception",
                    person_index=i,
                    bbox=person_bboxes[i],
                    error=str(result)
                )
                face_analyses.append(None)
            else:
                face_analyses.append(result)

        return face_analyses

    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """Get performance statistics."""
        if self.analysis_count == 0:
            return {'analysis_count': 0}

        avg_time = self.total_analysis_time / self.analysis_count

        return {
            'analysis_count': self.analysis_count,
            'total_analysis_time': round(self.total_analysis_time, 3),
            'average_analysis_time_ms': round(avg_time * 1000, 2),
            'target_time_ms': self.settings.testing.performance_benchmarks.face_analysis_ms
        }

    async def cleanup(self) -> None:
        """Cleanup MediaPipe resources."""
        logger.info("cleaning_up_mediapipe_analyzer")

        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None

        if self.face_detection:
            self.face_detection.close()
            self.face_detection = None

        self.models_loaded = False
        logger.info("mediapipe_analyzer_cleanup_completed")

# Convenience function
async def analyze_face_orientation(
        frame: np.ndarray,
        person_bbox: Tuple[int, int, int, int]
) -> Optional[FaceAnalysis]:
    analyzer = MediaPipeFaceAnalyzer()
    await analyzer.load_models()

    try:
        return await analyzer.analyze_face_orientation(frame, person_bbox)
    finally:
        await analyzer.cleanup()









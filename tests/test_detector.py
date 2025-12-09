import pytest
import numpy as np
import cv2
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import time
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mipcs.vision.detector import YOLOPoseDetector, PersonDetection
from mipcs.utils.exceptions import ModelLoadError, ModelInferenceError


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 300), (128, 64, 0), -1)
    cv2.rectangle(frame, (400, 50), (500, 250), (64, 128, 0), -1)
    return frame


class TestYOLOPoseDetector:
    """Test suite for YOLOPoseDetector class."""

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Test successful detector initialization."""
        detector = YOLOPoseDetector()

        assert detector is not None
        assert detector.model_loaded is False
        assert detector.model is None

    @pytest.mark.asyncio
    async def test_model_loading_success(self):
        """Test successful model loading."""
        with patch('mipcs.vision.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model

            detector = YOLOPoseDetector()
            await detector.load_model()

            assert detector.model_loaded is True
            assert detector.model is not None

    @pytest.mark.asyncio
    async def test_model_loading_failure(self):
        """Test model loading failure handling."""
        with patch('mipcs.vision.detector.YOLO', side_effect=Exception("Model load failed")):
            detector = YOLOPoseDetector()

            with pytest.raises(ModelLoadError):
                await detector.load_model()

    @pytest.mark.asyncio
    async def test_detection_with_valid_frame(self, sample_frame):
        """Test person detection with valid input frame."""
        with patch('mipcs.vision.detector.YOLO') as mock_yolo:
            # Mock model and results
            mock_model = Mock()
            mock_result = Mock()

            # Mock boxes
            mock_boxes = Mock()
            mock_boxes.cls = np.array([0, 0])  # Person class
            mock_box_data = [Mock(), Mock()]
            mock_xyxy_0 = Mock()
            mock_xyxy_0.cpu.return_value.numpy.return_value = np.array([100, 100, 200, 300])
            mock_box_data[0].xyxy = [mock_xyxy_0]
            mock_box_data[0].conf = [0.85]
            mock_xyxy_1 = Mock()
            mock_xyxy_1.cpu.return_value.numpy.return_value = np.array([400, 50, 500, 250])
            mock_box_data[1].xyxy = [mock_xyxy_1]
            mock_box_data[1].conf = [0.75]
            mock_boxes.__getitem__ = lambda self, key: mock_box_data[key] if isinstance(key, int) else mock_boxes
            mock_boxes.__len__ = lambda self: 2
            mock_result.boxes = mock_boxes

            # Mock keypoints
            mock_keypoints = Mock()
            mock_kpt_data = [Mock(), Mock()]
            mock_xy_0 = Mock()
            mock_xy_0.cpu.return_value.numpy.return_value = np.random.random((17, 2))
            mock_kpt_data[0].xy = [mock_xy_0]
            mock_conf_0 = Mock()
            mock_conf_0.cpu.return_value.numpy.return_value = np.random.random(17)
            mock_kpt_data[0].conf = [mock_conf_0]

            mock_xy_1 = Mock()
            mock_xy_1.cpu.return_value.numpy.return_value = np.random.random((17, 2))
            mock_kpt_data[1].xy = [mock_xy_1]
            mock_conf_1 = Mock()
            mock_conf_1.cpu.return_value.numpy.return_value = np.random.random(17)
            mock_kpt_data[1].conf = [mock_conf_1]
            mock_keypoints.__getitem__ = lambda self, key: mock_kpt_data[key] if isinstance(key,
                                                                                            int) else mock_keypoints
            mock_keypoints.__len__ = lambda self: 2
            mock_result.keypoints = mock_keypoints

            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model

            detector = YOLOPoseDetector()
            await detector.load_model()
            detections = await detector.detect_async(sample_frame)

            assert len(detections) == 2
            assert all(isinstance(det, PersonDetection) for det in detections)
            assert all(det.confidence >= 0.6 for det in detections)

    @pytest.mark.asyncio
    async def test_detection_with_no_people(self):
        """Test detection when no people are present."""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('mipcs.vision.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.cls = np.array([])
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model

            detector = YOLOPoseDetector()
            await detector.load_model()
            detections = await detector.detect_async(empty_frame)

            assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_invalid_input_handling(self):
        """Test handling of invalid input frames."""
        with patch('mipcs.vision.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model

            detector = YOLOPoseDetector()
            await detector.load_model()

            with pytest.raises(ModelInferenceError):
                await detector.detect_async(None)

            with pytest.raises(ModelInferenceError):
                await detector.detect_async(np.array([]))

    @pytest.mark.asyncio
    async def test_performance_stats(self, sample_frame):
        """Test performance statistics tracking."""
        with patch('mipcs.vision.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.cls = np.array([])
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model

            detector = YOLOPoseDetector()
            await detector.load_model()

            # Run detection
            await detector.detect_async(sample_frame)

            stats = detector.get_performance_stats()

            assert 'inference_count' in stats
            assert stats['inference_count'] == 1
            assert 'average_inference_time_ms' in stats
            assert 'fps_capability' in stats


class TestPersonDetection:
    """Test suite for PersonDetection data class."""

    def test_person_detection_creation(self):
        """Test PersonDetection object creation."""
        bbox = (100, 100, 200, 300)
        confidence = 0.85
        keypoints = np.random.random((17, 2))
        keypoint_conf = np.random.random(17)

        detection = PersonDetection(
            bbox=bbox,
            confidence=confidence,
            pose_keypoints=keypoints,
            pose_confidence=keypoint_conf
        )

        assert detection.bbox == bbox
        assert detection.confidence == confidence
        assert detection.width == 100
        assert detection.height == 200
        assert detection.area == 20000

    def test_person_detection_center_calculation(self):
        """Test center point calculation."""
        bbox = (100, 100, 200, 300)
        keypoints = np.random.random((17, 2))
        keypoint_conf = np.random.random(17)

        detection = PersonDetection(
            bbox=bbox,
            confidence=0.85,
            pose_keypoints=keypoints,
            pose_confidence=keypoint_conf
        )

        expected_center = (150.0, 250.0)
        assert detection.center == expected_center

    def test_person_detection_to_dict(self):
        """Test conversion to dictionary."""
        bbox = (100, 100, 200, 300)
        keypoints = np.random.random((17, 2))
        keypoint_conf = np.random.random(17)

        detection = PersonDetection(
            bbox=bbox,
            confidence=0.85,
            pose_keypoints=keypoints,
            pose_confidence=keypoint_conf,
            track_id=1
        )

        result_dict = detection.to_dict()

        assert 'bbox' in result_dict
        assert 'confidence' in result_dict
        assert 'pose_keypoints' in result_dict
        assert 'pose_confidence' in result_dict
        assert 'track_id' in result_dict
        assert 'center' in result_dict
        assert 'width' in result_dict
        assert 'height' in result_dict


class TestDetectorIntegration:
    """Integration tests for detector with real model."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_model_loading(self):
        """Test loading actual YOLO11 model (requires model file)."""
        model_path = Path("models/yolo11n-pose.engine")
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found")

        detector = YOLOPoseDetector()
        await detector.load_model()

        assert detector.model_loaded is True
        assert detector.model is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_detection(self, sample_frame):
        """Test complete detection pipeline with real model."""
        model_path = Path("models/yolo11n-pose.engine")
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found")

        detector = YOLOPoseDetector()
        await detector.load_model()

        detections = await detector.detect_async(sample_frame)

        assert isinstance(detections, list)
        for detection in detections:
            assert isinstance(detection, PersonDetection)
            assert hasattr(detection, 'bbox')
            assert hasattr(detection, 'confidence')
            assert hasattr(detection, 'pose_keypoints')
            assert hasattr(detection, 'pose_confidence')

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, sample_frame):
        """Test detection performance meets requirements."""
        model_path = Path("models/yolo11n-pose.engine")
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found")

        detector = YOLOPoseDetector()
        await detector.load_model()

        # Run multiple detections
        for _ in range(10):
            await detector.detect_async(sample_frame)

        stats = detector.get_performance_stats()

        # Should meet performance targets
        assert stats['inference_count'] == 10
        assert stats['average_inference_time_ms'] < 50  # Should be under 50ms
        assert stats['fps_capability'] > 20  # Should support at least 20 FPS
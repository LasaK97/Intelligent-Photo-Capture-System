import pytest
import numpy as np
import cv2
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import time
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mipcs.vision.detector import YOLOPoseDetector
from config.settings import Settings
from mipcs.utils.exceptions import ModelLoadError, DetectionError

@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = Mock(spec=Settings)
    config.vision = Mock()
    config.vision.yolo_model_path = "models/yolo11n-pose.pt"
    config.vision.confidence_threshold = 0.6
    config.vision.device = "cuda"
    config.vision.max_detections = 6
    config.vision.image_size = 640
    config.performance = Mock()
    config.performance.max_inference_time_ms = 50.0
    return config


@pytest.fixture
def sample_frame():
    """Create a sample frame with people for testing."""
    # Create a 640x480 RGB frame with simulated people
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some rectangular regions to simulate people
    cv2.rectangle(frame, (100, 100), (200, 300), (128, 64, 0), -1)
    cv2.rectangle(frame, (400, 50), (500, 250), (64, 128, 0), -1)

    return frame


@pytest.fixture
async def detector(mock_config):
    """Create detector instance for testing."""
    with patch('ultralytics.YOLO') as mock_yolo:
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        detector = YOLOPoseDetector(mock_config)
        await detector.initialize()
        yield detector


class TestYOLOPoseDetector:
    """Test suite for YOLOPoseDetector class."""

    @pytest.mark.asyncio
    async def test_initialization_success(self, mock_config):
        """Test successful detector initialization."""
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model

            detector = YOLOPoseDetector(mock_config)
            await detector.initialize()

            assert detector.initialized
            assert detector.model is not None
            mock_yolo.assert_called_once_with(mock_config.vision.yolo_model_path)

    @pytest.mark.asyncio
    async def test_initialization_failure(self, mock_config):
        """Test detector initialization failure handling."""
        with patch('ultralytics.YOLO', side_effect=Exception("Model load failed")):
            detector = YOLOPoseDetector(mock_config)

            with pytest.raises(ModelLoadError):
                await detector.initialize()

    @pytest.mark.asyncio
    async def test_detection_with_valid_frame(self, detector, sample_frame):
        """Test person detection with valid input frame."""
        # Mock YOLO prediction results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.data = np.array([
            [100, 100, 200, 300, 0.85, 0],  # x1, y1, x2, y2, conf, class
            [400, 50, 500, 250, 0.75, 0]
        ])

        # Mock keypoints
        mock_result.keypoints = Mock()
        mock_result.keypoints.data = np.random.random((2, 17, 3))  # 2 people, 17 keypoints, x,y,conf

        detector.model.predict = Mock(return_value=[mock_result])

        detections = await detector.detect_persons_with_pose(sample_frame)

        assert len(detections) == 2
        assert all('bbox' in det for det in detections)
        assert all('pose_keypoints' in det for det in detections)
        assert all('confidence' >= 0.6 for det in detections)

    @pytest.mark.asyncio
    async def test_detection_with_no_people(self, detector):
        """Test detection when no people are present."""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock empty results
        mock_result = Mock()
        mock_result.boxes = None
        mock_result.keypoints = None

        detector.model.predict = Mock(return_value=[mock_result])

        detections = await detector.detect_persons_with_pose(empty_frame)

        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_performance_timing(self, detector, sample_frame):
        """Test that detection meets performance requirements."""
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.data = np.array([[100, 100, 200, 300, 0.85, 0]])
        mock_result.keypoints = Mock()
        mock_result.keypoints.data = np.random.random((1, 17, 3))

        detector.model.predict = Mock(return_value=[mock_result])

        start_time = time.perf_counter()
        await detector.detect_persons_with_pose(sample_frame)
        inference_time = (time.perf_counter() - start_time) * 1000

        # Should complete within configured time limit
        assert inference_time < detector.config.performance.max_inference_time_ms

    @pytest.mark.asyncio
    async def test_confidence_filtering(self, detector, sample_frame):
        """Test that low-confidence detections are filtered out."""
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.data = np.array([
            [100, 100, 200, 300, 0.85, 0],  # High confidence - should keep
            [400, 50, 500, 250, 0.45, 0]  # Low confidence - should filter
        ])
        mock_result.keypoints = Mock()
        mock_result.keypoints.data = np.random.random((2, 17, 3))

        detector.model.predict = Mock(return_value=[mock_result])

        detections = await detector.detect_persons_with_pose(sample_frame)

        # Should only return high-confidence detection
        assert len(detections) == 1
        assert detections[0]['confidence'] >= detector.config.vision.confidence_threshold

    @pytest.mark.asyncio
    async def test_max_detections_limit(self, detector, sample_frame):
        """Test that maximum number of detections is respected."""
        # Create more detections than the limit
        num_detections = detector.config.vision.max_detections + 2
        mock_boxes = np.array([
            [i * 50, 100, (i + 1) * 50, 300, 0.85, 0]
            for i in range(num_detections)
        ])

        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.data = mock_boxes
        mock_result.keypoints = Mock()
        mock_result.keypoints.data = np.random.random((num_detections, 17, 3))

        detector.model.predict = Mock(return_value=[mock_result])

        detections = await detector.detect_persons_with_pose(sample_frame)

        assert len(detections) <= detector.config.vision.max_detections

    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, detector):
        """Test handling of invalid input frames."""
        with pytest.raises(DetectionError):
            await detector.detect_persons_with_pose(None)

        with pytest.raises(DetectionError):
            await detector.detect_persons_with_pose(np.array([]))

    def test_pose_quality_calculation(self, detector):
        """Test pose quality assessment logic."""
        # Mock keypoint data with varying confidence
        keypoints = np.array([
            [100, 200, 0.9],  # High confidence
            [150, 220, 0.8],  # High confidence
            [120, 180, 0.3],  # Low confidence
            [0, 0, 0.0],  # Invalid keypoint
        ])

        quality = detector._calculate_pose_quality(keypoints)

        assert 0.0 <= quality <= 1.0
        # Quality should reflect the mix of good and poor keypoints
        assert 0.4 < quality < 0.8


class TestDetectorIntegration:
    """Integration tests for detector with real model loading."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_model_loading(self, mock_config):
        """Test loading actual YOLO11 model (requires model file)."""
        # Skip if model file doesn't exist
        model_path = Path(mock_config.vision.yolo_model_path)
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found")

        detector = YOLOPoseDetector(mock_config)
        await detector.initialize()

        assert detector.initialized
        assert detector.model is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_detection(self, mock_config, sample_frame):
        """Test complete detection pipeline with real model."""
        model_path = Path(mock_config.vision.yolo_model_path)
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found")

        detector = YOLOPoseDetector(mock_config)
        await detector.initialize()

        detections = await detector.detect_persons_with_pose(sample_frame)

        # Should return valid detection structure
        assert isinstance(detections, list)
        for detection in detections:
            assert 'bbox' in detection
            assert 'confidence' in detection
            assert 'pose_keypoints' in detection
            assert 'pose_confidence' in detection
            assert 'pose_quality' in detection
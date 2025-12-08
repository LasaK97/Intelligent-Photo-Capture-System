import pytest
import numpy as np
import cv2
import asyncio
from unittest.mock import Mock, patch

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


from mipcs.vision.face_analyzer import MediaPipeFaceAnalyzer
from config.settings import Settings
from mipcs.utils.exceptions import FaceAnalysisError


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = Mock(spec=Settings)
    config.vision = Mock()
    config.vision.face_detection_confidence = 0.7
    config.vision.face_mesh_confidence = 0.5
    config.vision.max_faces = 4
    config.vision.frontal_face_threshold = 30.0  # degrees
    return config


@pytest.fixture
def face_analyzer(mock_config):
    """Create face analyzer instance."""
    with patch('mediapipe.solutions.face_mesh.FaceMesh'):
        analyzer = MediaPipeFaceAnalyzer(mock_config)
        yield analyzer


@pytest.fixture
def sample_face_frame():
    """Create a sample frame with a face-like region."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple face-like shape
    cv2.circle(frame, (320, 240), 80, (200, 180, 160), -1)
    # Eyes
    cv2.circle(frame, (300, 220), 10, (50, 50, 50), -1)
    cv2.circle(frame, (340, 220), 10, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(frame, (320, 260), (25, 15), 0, 0, 180, (100, 50, 50), -1)
    return frame


class TestMediaPipeFaceAnalyzer:
    """Test suite for MediaPipeFaceAnalyzer class."""

    def test_initialization(self, mock_config):
        """Test face analyzer initialization."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh:
            analyzer = MediaPipeFaceAnalyzer(mock_config)

            assert analyzer.config == mock_config
            mock_face_mesh.assert_called_once()

    @pytest.mark.asyncio
    async def test_face_analysis_with_detection(self, face_analyzer, sample_face_frame):
        """Test face analysis when face is detected."""
        # Mock MediaPipe detection results
        mock_results = Mock()
        mock_results.multi_face_landmarks = [Mock()]

        # Mock landmark coordinates for frontal face
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0

        mock_results.multi_face_landmarks[0].landmark = [mock_landmark] * 468

        face_analyzer.face_mesh.process = Mock(return_value=mock_results)

        person_bbox = [200, 100, 440, 380]  # x1, y1, x2, y2
        result = await face_analyzer.analyze_face_orientation(sample_face_frame, person_bbox)

        assert result['landmarks_detected'] is True
        assert 'facing_camera' in result
        assert 'orientation' in result
        assert 'confidence' in result

    @pytest.mark.asyncio
    async def test_face_analysis_no_detection(self, face_analyzer):
        """Test face analysis when no face is detected."""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock no detection results
        mock_results = Mock()
        mock_results.multi_face_landmarks = None

        face_analyzer.face_mesh.process = Mock(return_value=mock_results)

        person_bbox = [200, 100, 440, 380]
        result = await face_analyzer.analyze_face_orientation(empty_frame, person_bbox)

        assert result['landmarks_detected'] is False
        assert result['facing_camera'] is False
        assert result['orientation'] == 'unknown'
        assert result['confidence'] == 0.0

    @pytest.mark.asyncio
    async def test_invalid_bbox_handling(self, face_analyzer, sample_face_frame):
        """Test handling of invalid bounding boxes."""
        invalid_bboxes = [
            None,
            [],
            [0, 0, 0, 0],  # Zero area
            [100, 100, 50, 50],  # Negative dimensions
            [-50, -50, 100, 100]  # Partially out of bounds
        ]

        for bbox in invalid_bboxes:
            with pytest.raises(FaceAnalysisError):
                await face_analyzer.analyze_face_orientation(sample_face_frame, bbox)

    def test_extract_face_crop(self, face_analyzer, sample_face_frame):
        """Test face region extraction from frame."""
        person_bbox = [200, 100, 440, 380]

        face_crop = face_analyzer._extract_face_crop(sample_face_frame, person_bbox)

        expected_height = 380 - 100  # bbox height
        expected_width = 440 - 200  # bbox width

        assert face_crop.shape[:2] == (expected_height, expected_width)
        assert face_crop.dtype == np.uint8

    def test_orientation_classification(self, face_analyzer):
        """Test face orientation classification logic."""
        # Test frontal face
        frontal_angles = [0.0, 10.0, -8.0, 15.0]
        for angle in frontal_angles:
            orientation = face_analyzer._classify_orientation(angle)
            assert orientation == 'frontal'

        # Test profile face
        profile_angles = [45.0, -60.0, 90.0, -90.0]
        for angle in profile_angles:
            orientation = face_analyzer._classify_orientation(angle)
            assert orientation == 'profile'

        # Test partial profile
        partial_angles = [25.0, -35.0, 40.0, -25.0]
        for angle in partial_angles:
            orientation = face_analyzer._classify_orientation(angle)
            assert orientation == 'partial_profile'

    def test_facing_camera_determination(self, face_analyzer):
        """Test logic for determining if face is facing camera."""
        # Frontal orientations should be facing camera
        assert face_analyzer._is_facing_camera('frontal') is True

        # Profile orientations should not be facing camera
        assert face_analyzer._is_facing_camera('profile') is False

        # Partial profile depends on configuration
        facing = face_analyzer._is_facing_camera('partial_profile')
        assert isinstance(facing, bool)

    def test_calculate_head_pose_angle(self, face_analyzer):
        """Test head pose angle calculation from landmarks."""
        # Mock landmark data for different poses
        frontal_landmarks = np.array([
            [0.3, 0.3, 0.0],  # Left eye corner
            [0.7, 0.3, 0.0],  # Right eye corner
            [0.5, 0.7, 0.0]  # Nose tip
        ])

        angle = face_analyzer._calculate_head_pose_angle(frontal_landmarks)
        assert isinstance(angle, float)
        assert -180.0 <= angle <= 180.0

    @pytest.mark.asyncio
    async def test_multiple_faces_handling(self, face_analyzer, sample_face_frame):
        """Test handling of multiple faces in single detection."""
        # Mock multiple face detection
        mock_results = Mock()
        mock_results.multi_face_landmarks = [Mock(), Mock()]  # Two faces

        # Mock landmark data for both faces
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0

        for face_landmarks in mock_results.multi_face_landmarks:
            face_landmarks.landmark = [mock_landmark] * 468

        face_analyzer.face_mesh.process = Mock(return_value=mock_results)

        person_bbox = [200, 100, 440, 380]
        result = await face_analyzer.analyze_face_orientation(sample_face_frame, person_bbox)

        # Should process the first/primary face detected
        assert result['landmarks_detected'] is True


class TestFaceAnalyzerIntegration:
    """Integration tests for face analyzer."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_face_detection(self, mock_config, sample_face_frame):
        """Test with real MediaPipe model (requires MediaPipe installation)."""
        try:
            import mediapipe
        except ImportError:
            pytest.skip("MediaPipe not installed")

        analyzer = MediaPipeFaceAnalyzer(mock_config)

        person_bbox = [200, 100, 440, 380]
        result = await analyzer.analyze_face_orientation(sample_face_frame, person_bbox)

        # Should return valid result structure
        assert 'landmarks_detected' in result
        assert 'facing_camera' in result
        assert 'orientation' in result
        assert 'confidence' in result
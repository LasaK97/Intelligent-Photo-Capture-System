import pytest
import numpy as np
import cv2
import asyncio
from unittest.mock import Mock, patch

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mipcs.vision.face_analyzer import MediaPipeFaceAnalyzer, FaceAnalysis


@pytest.fixture
def sample_face_frame():
    """Create a sample frame with a face-like region."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 80, (200, 180, 160), -1)
    cv2.circle(frame, (300, 220), 10, (50, 50, 50), -1)
    cv2.circle(frame, (340, 220), 10, (50, 50, 50), -1)
    cv2.ellipse(frame, (320, 260), (25, 15), 0, 0, 180, (100, 50, 50), -1)
    return frame


class TestMediaPipeFaceAnalyzer:
    """Test suite for MediaPipeFaceAnalyzer class."""

    def test_initialization(self):
        """Test face analyzer initialization."""
        analyzer = MediaPipeFaceAnalyzer()

        assert analyzer is not None
        assert analyzer.models_loaded is False
        assert analyzer.face_mesh is None
        assert analyzer.face_detection is None

    @pytest.mark.asyncio
    async def test_model_loading(self):
        """Test MediaPipe model loading."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh'), \
                patch('mediapipe.solutions.face_detection.FaceDetection'):
            analyzer = MediaPipeFaceAnalyzer()
            await analyzer.load_models()

            assert analyzer.models_loaded is True

    @pytest.mark.asyncio
    async def test_face_analysis_with_detection(self, sample_face_frame):
        """Test face analysis when face is detected."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class, \
                patch('mediapipe.solutions.face_detection.FaceDetection'):
            # Mock MediaPipe detection results
            mock_face_mesh = Mock()
            mock_results = Mock()
            mock_results.multi_face_landmarks = [Mock()]

            # Mock landmark coordinates for frontal face
            mock_landmark = Mock()
            mock_landmark.x = 0.5
            mock_landmark.y = 0.5
            mock_landmark.z = 0.0

            mock_results.multi_face_landmarks[0].landmark = [mock_landmark] * 468
            mock_face_mesh.process = Mock(return_value=mock_results)
            mock_face_mesh_class.return_value = mock_face_mesh

            analyzer = MediaPipeFaceAnalyzer()
            await analyzer.load_models()

            person_bbox = (200, 100, 440, 380)
            result = await analyzer.analyze_face_orientation(sample_face_frame, person_bbox)

            assert result is not None
            assert isinstance(result, FaceAnalysis)
            assert result.bbox is not None
            assert result.facing_camera is not None
            assert result.orientation is not None

    @pytest.mark.asyncio
    async def test_face_analysis_no_detection(self):
        """Test face analysis when no face is detected."""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class, \
                patch('mediapipe.solutions.face_detection.FaceDetection'):
            mock_face_mesh = Mock()
            mock_results = Mock()
            mock_results.multi_face_landmarks = None
            mock_face_mesh.process = Mock(return_value=mock_results)
            mock_face_mesh_class.return_value = mock_face_mesh

            analyzer = MediaPipeFaceAnalyzer()
            await analyzer.load_models()

            person_bbox = (200, 100, 440, 380)
            result = await analyzer.analyze_face_orientation(empty_frame, person_bbox)

            # Should return None when no face detected
            assert result is None or (result.orientation == "profile" and result.confidence <= 0.5)

    @pytest.mark.asyncio
    async def test_performance_stats(self):
        """Test performance statistics tracking."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh'), \
                patch('mediapipe.solutions.face_detection.FaceDetection'):
            analyzer = MediaPipeFaceAnalyzer()
            await analyzer.load_models()

            stats = analyzer.get_performance_stats()

            assert 'analysis_count' in stats
            assert stats['analysis_count'] == 0

    @pytest.mark.asyncio
    async def test_multiple_faces_analysis(self, sample_face_frame):
        """Test analyzing multiple faces concurrently."""
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class, \
                patch('mediapipe.solutions.face_detection.FaceDetection'):
            mock_face_mesh = Mock()
            mock_results = Mock()
            mock_results.multi_face_landmarks = [Mock()]

            mock_landmark = Mock()
            mock_landmark.x = 0.5
            mock_landmark.y = 0.5
            mock_landmark.z = 0.0
            mock_results.multi_face_landmarks[0].landmark = [mock_landmark] * 468

            mock_face_mesh.process = Mock(return_value=mock_results)
            mock_face_mesh_class.return_value = mock_face_mesh

            analyzer = MediaPipeFaceAnalyzer()
            await analyzer.load_models()

            person_bboxes = [
                (100, 100, 200, 300),
                (400, 100, 500, 300)
            ]

            results = await analyzer.analyze_multiple_faces(sample_face_frame, person_bboxes)

            assert len(results) == 2


class TestFaceAnalysis:
    """Test suite for FaceAnalysis data class."""

    def test_face_analysis_creation(self):
        """Test FaceAnalysis object creation."""
        bbox = (100, 100, 200, 300)

        analysis = FaceAnalysis(
            bbox=bbox,
            facing_camera=True,
            orientation="frontal",
            landmarks=None,
            confidence=0.85,
            analysis_metrics={'method': 'face_mesh'}
        )

        assert analysis.bbox == bbox
        assert analysis.facing_camera is True
        assert analysis.orientation == "frontal"
        assert analysis.confidence == 0.85

    def test_face_analysis_to_dict(self):
        """Test conversion to dictionary."""
        bbox = (100, 100, 200, 300)

        analysis = FaceAnalysis(
            bbox=bbox,
            facing_camera=True,
            orientation="frontal",
            landmarks=None,
            confidence=0.85,
            analysis_metrics={'method': 'face_mesh'}
        )

        result_dict = analysis.to_dict()

        assert 'bbox' in result_dict
        assert 'facing_camera' in result_dict
        assert 'orientation' in result_dict
        assert 'confidence' in result_dict
        assert 'analysis_metrics' in result_dict


class TestFaceAnalyzerIntegration:
    """Integration tests for face analyzer."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_face_detection(self, sample_face_frame):
        """Test with real MediaPipe model (requires MediaPipe installation)."""
        try:
            import mediapipe
        except ImportError:
            pytest.skip("MediaPipe not installed")

        analyzer = MediaPipeFaceAnalyzer()
        await analyzer.load_models()

        person_bbox = (200, 100, 440, 380)
        result = await analyzer.analyze_face_orientation(sample_face_frame, person_bbox)

        # Should return valid result structure or None
        if result is not None:
            assert isinstance(result, FaceAnalysis)
            assert result.bbox is not None
            assert result.facing_camera is not None
            assert result.orientation is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, sample_face_frame):
        """Test face analysis performance."""
        try:
            import mediapipe
        except ImportError:
            pytest.skip("MediaPipe not installed")

        analyzer = MediaPipeFaceAnalyzer()
        await analyzer.load_models()

        person_bbox = (200, 100, 440, 380)

        # Run multiple analyses
        for _ in range(10):
            await analyzer.analyze_face_orientation(sample_face_frame, person_bbox)

        stats = analyzer.get_performance_stats()

        assert 'analysis_count' in stats
        if stats['analysis_count'] > 0:
            assert 'average_analysis_time_ms' in stats
            # Face analysis should be reasonably fast
            assert stats['average_analysis_time_ms'] < 100

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup of MediaPipe resources."""
        try:
            import mediapipe
        except ImportError:
            pytest.skip("MediaPipe not installed")

        analyzer = MediaPipeFaceAnalyzer()
        await analyzer.load_models()

        assert analyzer.models_loaded is True

        await analyzer.cleanup()

        assert analyzer.models_loaded is False
        assert analyzer.face_mesh is None
        assert analyzer.face_detection is None
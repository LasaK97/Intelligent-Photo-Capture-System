from typing import Optional, Any, Dict

class ManriixError(Exception):
    """Base exception for all Manriix system errors"""

    def __init__(
            self,
            message,
            error_code: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
            original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception into a dictionary."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'original_error': str(self.original_error) if self.original_error else None
        }

class VisionError(ManriixError):
    """Base class for vision-related errors."""
    pass

class DetectionError(VisionError):
    """YOLO detection related errors."""
    pass

class FaceAnalysisError(VisionError):
    """MediaPipe face analysis related errors."""
    pass

class DepthProcessingError(VisionError):
    """RealSense depth processing related errors."""
    pass

class FrameClientError(VisionError):
    """Canon frame client related errors."""
    pass

class ConfigurationError(ManriixError):
    """Configuration and settings related errors."""
    pass

class HardwareError(ManriixError):
    """Hardware interface related errors."""
    pass

class PerformanceError(ManriixError):
    """Performance and resource related errors."""
    pass

class NetworkError(ManriixError):
    """Network communication related errors."""
    pass

class TimeoutError(ManriixError):
    """Timeout related errors."""
    pass

class ModelLoadError(DetectionError):
    """Error loading AI models."""
    pass

class ModelInferenceError(DetectionError):
    """Error during model inference."""
    pass

class CameraConnectionError(FrameClientError):
    """Error connecting to camera."""
    pass

class FrameProcessingError(VisionError):
    """Error processing camera frames."""
    pass

class DepthDataError(DepthProcessingError):
    """Error with depth data processing."""
    pass

class CalibrationError(VisionError):
    """Camera calibration related errors."""
    pass

class ControlError(ManriixError):
    """Base class for control system errors."""
    pass


class GimbalError(ControlError):
    """Gimbal control related errors."""
    pass


class FocusError(ControlError):
    """Focus motor related errors."""
    pass


class PositioningError(ManriixError):
    """3D positioning related errors."""
    pass


class SceneClassificationError(ManriixError):
    """Scene analysis related errors."""
    pass


class StateMachineError(ManriixError):
    """State machine execution errors."""
    pass


class GimbalTimeoutError(GimbalError):
    """Gimbal movement timeout."""
    pass


class FocusCalibrationError(FocusError):
    """Focus calibration data error."""
    pass


class PositionCalculationError(PositioningError):
    """3D position calculation error."""
    pass


class DepthInvalidError(PositioningError):
    """Invalid or missing depth data."""
    pass


class SceneAnalysisError(SceneClassificationError):
    """Scene analysis computation error."""
    pass
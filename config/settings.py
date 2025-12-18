
from typing import Dict, Any, Optional
from pathlib import Path

# Import domain loader
from config.domain_loader import DomainConfigLoader


class DictWrapper:
    """
    Wrapper that provides attribute-style access to dictionaries
    For backwards compatibility with old Pydantic models
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        if name not in self._data:
            raise AttributeError(f"Config has no attribute '{name}'")

        value = self._data[name]

        # If value is a dict, wrap it for nested access
        if isinstance(value, dict):
            return DictWrapper(value)

        # If value is a list of dicts, wrap each dict
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return [DictWrapper(item) if isinstance(item, dict) else item
                    for item in value]

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access as well"""
        value = self._data[key]
        if isinstance(value, dict):
            return DictWrapper(value)
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style get with default"""
        return self._data.get(key, default)

    def keys(self):
        """Return keys like a dictionary"""
        return self._data.keys()

    def values(self):
        """Return values like a dictionary"""
        return self._data.values()

    def items(self):
        """Return items like a dictionary"""
        return self._data.items()

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to plain dictionary"""
        return self._data

    def model_dump(self) -> Dict[str, Any]:
        """Pydantic V2 compatibility - alias for to_dict()"""
        return self.to_dict()


class UnifiedSettings:
    """
    Unified settings that provide backwards-compatible access to
    domain-based configuration system.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize unified settings

        Args:
            config_dir: Path to config directory (defaults to this file's parent)
        """
        # Initialize domain loader
        self._loader = DomainConfigLoader(config_dir)

        # Create wrapped versions for backwards compatibility
        self._hardware_wrapper = DictWrapper(self._loader.get_hardware_config())
        self._algorithms_wrapper = DictWrapper(self._loader.get_algorithms_config())
        self._workflows_wrapper = DictWrapper(self._loader.get_workflows_config())
        self._system_wrapper = DictWrapper(self._loader.get_system_config())
        self._shared_wrapper = DictWrapper(self._loader.get_shared_config())

    # =========================================================================
    # Domain-level accessors (NEW recommended style)
    # =========================================================================

    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware domain configuration"""
        return self._loader.get_hardware_config()

    def get_algorithms_config(self) -> Dict[str, Any]:
        """Get algorithms domain configuration"""
        return self._loader.get_algorithms_config()

    def get_workflows_config(self) -> Dict[str, Any]:
        """Get workflows domain configuration"""
        return self._loader.get_workflows_config()

    def get_system_config(self) -> Dict[str, Any]:
        """Get system domain configuration"""
        return self._loader.get_system_config()

    def get_shared_config(self) -> Dict[str, Any]:
        """Get shared configuration"""
        return self._loader.get_shared_config()

    # =========================================================================
    # Specific config accessors (NEW recommended style)
    # =========================================================================

    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return self._loader.get_camera_config()

    def get_gimbal_config(self) -> Dict[str, Any]:
        """Get gimbal configuration"""
        return self._loader.get_gimbal_config()

    def get_vision_config(self) -> Dict[str, Any]:
        """Get vision configuration"""
        return self._loader.get_vision_config()

    def get_positioning_config(self) -> Dict[str, Any]:
        """Get positioning configuration"""
        return self._loader.get_positioning_config()

    def get_auto_framing_config(self) -> Dict[str, Any]:
        """Get auto-framing configuration"""
        return self._loader.get_auto_framing_config()

    def get_photo_capture_workflow(self) -> Dict[str, Any]:
        """Get photo capture workflow configuration"""
        workflows = self._loader.get_workflows_config()
        return workflows.get('photo_capture', {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self._loader.get_performance_config()

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self._loader.get_logging_config()

    def get_ros2_config(self) -> Dict[str, Any]:
        """Get ROS2 configuration"""
        return self._loader.get_ros2_config()

    # =========================================================================
    # Backwards-compatible attribute access (OLD style - for migration)
    # =========================================================================

    @property
    def hardware(self) -> DictWrapper:
        """Backwards compatible: settings.hardware.camera.exposure_control"""
        return self._hardware_wrapper

    @property
    def algorithms(self) -> DictWrapper:
        """Backwards compatible: settings.algorithms.vision.yolo"""
        return self._algorithms_wrapper

    @property
    def workflows(self) -> DictWrapper:
        """Backwards compatible: settings.workflows.photo_capture"""
        return self._workflows_wrapper

    @property
    def system(self) -> DictWrapper:
        """Backwards compatible: settings.system.performance"""
        return self._system_wrapper

    @property
    def shared(self) -> DictWrapper:
        """Backwards compatible: settings.shared.hardware.camera"""
        return self._shared_wrapper

    # =========================================================================
    # Legacy attribute mappings (OLD monolithic config.yaml style)
    # =========================================================================

    @property
    def vision(self) -> DictWrapper:
        """Legacy: settings.vision.yolo"""
        vision_config = self._loader.get_vision_config()
        camera_config = self._loader.get_camera_config()
        vision_config['canon'] = camera_config.get('frame_client', {})
        sensors_config = self._loader.get_hardware_config().get('sensors', {})
        vision_config['realsense'] = sensors_config.get('realsense', {})
        return DictWrapper(vision_config)

    @property
    def camera_control(self) -> DictWrapper:
        """Legacy: settings.camera_control.gimbal"""
        gimbal_data = self._loader.get_gimbal_config()
        sensors_data = self._loader.get_hardware_config().get('sensors', {})
        return DictWrapper({
            'gimbal': gimbal_data.get('gimbal_control', {}),
            'focus': sensors_data.get('focus', {})
        })

    @property
    def positioning(self) -> DictWrapper:
        """Legacy: settings.positioning - with flattened coordinate_system fields"""
        pos_config = self._loader.get_positioning_config().copy()

        # Flatten coordinate_system fields to root level for backwards compatibility
        if 'coordinate_system' in pos_config:
            coord_sys = pos_config['coordinate_system']

            # Add root-level access to nested fields
            if 'use_dynamic_transforms' in coord_sys:
                pos_config['use_dynamic_transforms'] = coord_sys['use_dynamic_transforms']

            if 'transform_cache_duration' in coord_sys:
                pos_config['transform_cache_duration'] = coord_sys['transform_cache_duration']

        return DictWrapper(pos_config)

    @property
    def photo_capture(self) -> DictWrapper:
        """Legacy: settings.photo_capture"""
        return DictWrapper(self.get_photo_capture_workflow())

    @property
    def performance(self) -> DictWrapper:
        """Legacy: settings.performance"""
        return DictWrapper(self._loader.get_performance_config())

    @property
    def logging(self) -> DictWrapper:
        """Legacy: settings.logging"""
        return DictWrapper(self._loader.get_logging_config())

    @property
    def ros2_topics(self) -> DictWrapper:
        """Legacy: settings.ros2_topics - flattened for backwards compatibility"""
        shared_topics = self._loader.shared_loader.ros_topics

        # Flatten nested structure: input_topics, output_topics, status_topics → root level
        flattened = {}

        if 'input_topics' in shared_topics:
            flattened.update(shared_topics['input_topics'])

        if 'output_topics' in shared_topics:
            flattened.update(shared_topics['output_topics'])

        if 'status_topics' in shared_topics:
            flattened.update(shared_topics['status_topics'])

        return DictWrapper(flattened)

    @property
    def control_timing(self) -> DictWrapper:
        """Legacy: settings.control_timing"""
        performance = self._loader.get_performance_config()
        return DictWrapper(performance.get('control_timing', {}))

    @property
    def threading(self) -> DictWrapper:
        """Legacy: settings.threading"""
        return DictWrapper(self._loader.get_performance_config().get('threading', {}))

    @property
    def testing(self) -> DictWrapper:
        """Legacy: settings.testing"""
        return DictWrapper(self._loader.get_performance_config().get('testing', {}))

    @property
    def ros2(self) -> DictWrapper:
        """Legacy: settings.ros2"""
        return DictWrapper(self._loader.get_ros2_config())

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> bool:
        """Validate all configurations"""
        return self._loader.validate_all_configs()


# =============================================================================
# Module-level singleton
# =============================================================================

_settings_instance: Optional[UnifiedSettings] = None


def get_settings(config_dir: Optional[Path] = None) -> UnifiedSettings:
    """
    Get or create the unified settings singleton

    Args:
        config_dir: Path to config directory (only used on first call)

    Returns:
        UnifiedSettings instance
    """
    global _settings_instance

    if _settings_instance is None:
        _settings_instance = UnifiedSettings(config_dir)

    return _settings_instance


def reset_settings():
    """Reset settings singleton (useful for testing)"""
    global _settings_instance
    _settings_instance = None


# =============================================================================
# Convenience functions (NEW recommended style)
# =============================================================================

def get_hardware_config() -> Dict[str, Any]:
    """Get hardware configuration"""
    return get_settings().get_hardware_config()


def get_algorithms_config() -> Dict[str, Any]:
    """Get algorithms configuration"""
    return get_settings().get_algorithms_config()


def get_workflows_config() -> Dict[str, Any]:
    """Get workflows configuration"""
    return get_settings().get_workflows_config()


def get_system_config() -> Dict[str, Any]:
    """Get system configuration"""
    return get_settings().get_system_config()


def get_shared_config() -> Dict[str, Any]:
    """Get shared configuration"""
    return get_settings().get_shared_config()


def get_camera_config() -> Dict[str, Any]:
    """Get camera configuration"""
    return get_settings().get_camera_config()


def get_gimbal_config() -> Dict[str, Any]:
    """Get gimbal configuration"""
    return get_settings().get_gimbal_config()


def get_vision_config() -> Dict[str, Any]:
    """Get vision configuration"""
    return get_settings().get_vision_config()


def get_positioning_config() -> Dict[str, Any]:
    """Get positioning configuration"""
    return get_settings().get_positioning_config()


def get_auto_framing_config() -> Dict[str, Any]:
    """Get auto-framing configuration"""
    return get_settings().get_auto_framing_config()


def get_photo_capture_workflow() -> Dict[str, Any]:
    """Get photo capture workflow configuration"""
<<<<<<< Updated upstream
    return get_settings().get_photo_capture_workflow()


def get_workflow_config() -> Dict[str, Any]:
    """Get photo capture workflow configuration (alias)"""
    return get_photo_capture_workflow()


def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration"""
    return get_settings().get_performance_config()


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration"""
    return get_settings().get_logging_config()


def get_ros2_config() -> Dict[str, Any]:
    """Get ROS2 configuration"""
    return get_settings().get_ros2_config()
=======
    return get_settings().workflows.photo_capture.to_dict()

def get_workflow_config() -> Dict[str, Any]:
    """Get photo capture workflow configuration"""
    return get_settings().get_workflows_config()['photo_capture']


def get_photo_capture_workflow() -> Dict[str, Any]:
    """Get photo capture workflow configuration (alias)"""
    return get_workflow_config()
>>>>>>> Stashed changes

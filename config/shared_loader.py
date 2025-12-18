import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import re


class SharedConfigLoader:
    """Loader for shared configuration files"""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize shared config loader"""
        if config_dir is None:
            config_dir = Path(__file__).parent

        self.config_dir = Path(config_dir)
        self.shared_dir = self.config_dir / "shared"

        # Validate shared directory exists
        if not self.shared_dir.exists():
            raise FileNotFoundError(
                f"Shared config directory not found: {self.shared_dir}"
            )

        # Load all shared configurations
        self._hardware_specs = self._load_yaml("hardware_specs.yaml")
        self._coordinate_frames = self._load_yaml("coordinate_frames.yaml")
        self._ros_topics = self._load_yaml("ros_topics.yaml")
        self._defaults = self._load_yaml("defaults.yaml")

        # Create unified shared config dictionary for reference resolution
        self._shared_config = {
            "hardware": self._hardware_specs,
            "frames": self._coordinate_frames,
            "topics": self._ros_topics,
            "defaults": self._defaults
        }

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file from shared directory"""
        filepath = self.shared_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Shared config file not found: {filepath}")

        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filename}: {e}")

    @property
    def hardware_specs(self) -> Dict[str, Any]:
        """Get hardware specifications (camera, gimbal, sensors)"""
        return self._hardware_specs

    @property
    def coordinate_frames(self) -> Dict[str, Any]:
        """Get coordinate frame definitions"""
        return self._coordinate_frames

    @property
    def ros_topics(self) -> Dict[str, Any]:
        """Get ROS2 topic names"""
        return self._ros_topics

    @property
    def defaults(self) -> Dict[str, Any]:
        """Get system defaults"""
        return self._defaults

    @property
    def all_shared(self) -> Dict[str, Any]:
        """Get all shared configs as unified dictionary"""
        return self._shared_config

    def get_camera_mount_height(self) -> float:
        """Get camera mount height (frequently used)"""
        return self._hardware_specs["camera"]["mount_height"]

    def get_focal_length_range(self) -> tuple[float, float]:
        """Get lens focal length range (min, max)"""
        lens = self._hardware_specs["camera"]["lens"]
        return lens["focal_length"]["min"], lens["focal_length"]["max"]

    def get_gimbal_limits(self) -> Dict[str, Dict[str, float]]:
        """Get gimbal joint limits"""
        return self._hardware_specs["gimbal"]["limits"]

    def get_frame_names(self) -> Dict[str, str]:
        """Get coordinate frame names"""
        return self._coordinate_frames["frames"]

    def get_input_topics(self) -> Dict[str, str]:
        """Get ROS2 input topic names"""
        return self._ros_topics["input_topics"]

    def get_output_topics(self) -> Dict[str, str]:
        """Get ROS2 output topic names"""
        return self._ros_topics["output_topics"]

    # =========================================================================
    # Reference Resolution
    # =========================================================================

    def resolve_reference(self, reference: str) -> Any:
        """Resolve a reference like 'hardware.camera.mount_height' to its value"""
        parts = reference.split('.')
        value = self._shared_config

        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError) as e:
            raise KeyError(f"Invalid reference path: {reference}") from e

    def resolve_references_in_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively resolve all ${shared.path} references in a config dictionary"""
        if isinstance(config, dict):
            return {
                key: self.resolve_references_in_config(value)
                for key, value in config.items()
            }
        elif isinstance(config, list):
            return [self.resolve_references_in_config(item) for item in config]
        elif isinstance(config, str):
            # Check if it's a reference pattern: ${shared.path}
            match = re.match(r'\$\{shared\.(.+)\}', config)
            if match:
                reference_path = match.group(1)
                return self.resolve_reference(reference_path)
            return config
        else:
            return config

    def validate_shared_configs(self) -> bool:
        """Validate that all shared configs have required keys"""
        # Validate hardware specs
        required_hardware = ["camera", "gimbal", "focus", "sensors"]
        for key in required_hardware:
            if key not in self._hardware_specs:
                raise ValueError(f"Missing required hardware spec: {key}")

        # Validate camera specs
        camera = self._hardware_specs["camera"]
        if "mount_height" not in camera:
            raise ValueError("Missing camera.mount_height in hardware specs")
        if "lens" not in camera:
            raise ValueError("Missing camera.lens in hardware specs")

        # Validate coordinate frames
        if "frames" not in self._coordinate_frames:
            raise ValueError("Missing frames in coordinate_frames.yaml")

        # Validate ROS topics
        required_topic_groups = ["input_topics", "output_topics", "status_topics"]
        for group in required_topic_groups:
            if group not in self._ros_topics:
                raise ValueError(f"Missing {group} in ros_topics.yaml")

        # Validate defaults
        if "system" not in self._defaults:
            raise ValueError("Missing system in defaults.yaml")

        return True

    def get_config_summary(self) -> str:
        """Get a summary of loaded shared configurations"""
        return f"""
Shared Configuration Summary:
============================
Hardware Specs:
  - Camera mount height: {self.get_camera_mount_height()}m
  - Focal length range: {self.get_focal_length_range()}mm
  - Gimbal joints: {len(self._hardware_specs['gimbal']['joint_names'])}

Coordinate Frames:
  - Defined frames: {len(self.get_frame_names())}
  - Primary system: {self._coordinate_frames['coordinate_system']['primary']}

ROS2 Topics:
  - Input topics: {len(self.get_input_topics())}
  - Output topics: {len(self.get_output_topics())}

System:
  - Name: {self._defaults['system']['name']}
  - Version: {self._defaults['system']['version']}
  - Environment: {self._defaults['system']['environment']}
"""

_shared_loader_instance: Optional[SharedConfigLoader] = None


def get_shared_loader(config_dir: Optional[Path] = None) -> SharedConfigLoader:
    """Get or create the shared config loader singleton"""
    global _shared_loader_instance

    if _shared_loader_instance is None:
        _shared_loader_instance = SharedConfigLoader(config_dir)

    return _shared_loader_instance


def reset_shared_loader():
    """Reset the shared loader singleton"""
    global _shared_loader_instance
    _shared_loader_instance = None

def get_camera_specs() -> Dict[str, Any]:
    """Get camera specifications from shared config"""
    return get_shared_loader().hardware_specs["camera"]


def get_gimbal_specs() -> Dict[str, Any]:
    """Get gimbal specifications from shared config"""
    return get_shared_loader().hardware_specs["gimbal"]


def get_ros_topics() -> Dict[str, Any]:
    """Get all ROS topics from shared config"""
    return get_shared_loader().ros_topics


def resolve_reference(reference: str) -> Any:
    """Resolve a shared config reference"""
    return get_shared_loader().resolve_reference(reference)
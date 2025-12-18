import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from config.shared_loader import SharedConfigLoader
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))



try:
    from config.validators import (
        validate_hardware_config,
        validate_algorithms_config,
        validate_workflows_config,
        validate_system_config,
        validate_all_domains
    )

    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    print("Warning: Pydantic validators not available. Skipping validation.")


class DomainConfigLoader:
    """Loader for domain-specific configuration files"""

    def __init__(self, config_dir: Optional[Path] = None):

        if config_dir is None:
            config_dir = Path(__file__).parent

        self.config_dir = Path(config_dir)

        # Initialize shared config loader first
        self.shared_loader = SharedConfigLoader(config_dir)

        # Load all domain configurations
        self._hardware = self._load_domain("hardware")
        self._algorithms = self._load_domain("algorithms")
        self._workflows = self._load_domain("workflows")
        self._system = self._load_domain("system")

        # Resolve all references to shared configs
        self._resolve_all_references()

    def _load_domain(self, domain_name: str) -> Dict[str, Any]:
        """load all YAML files in a domain directory"""
        domain_dir = self.config_dir / domain_name

        if not domain_dir.exists():
            raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

        configs = {}

        # Load all .yaml files in domain directory (including subdirectories)
        for yaml_file in domain_dir.rglob("*.yaml"):
            # Get relative path from domain directory
            rel_path = yaml_file.relative_to(domain_dir)

            # Create nested key structure for subdirectories
            # e.g., auto_framing/core.yaml -> auto_framing.core
            key_parts = list(rel_path.parts[:-1]) + [rel_path.stem]

            # Load YAML
            try:
                with open(yaml_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {yaml_file}: {e}")

            # Build nested structure
            if len(key_parts) == 1:
                # Top-level file: hardware/camera.yaml -> configs['camera']
                configs[key_parts[0]] = config_data
            else:
                # Nested file: algorithms/auto_framing/core.yaml
                # -> configs['auto_framing']['core']
                current = configs
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[key_parts[-1]] = config_data

        return configs

    def _resolve_all_references(self):
        """resolve all ${shared.path} references in domain configs"""
        self._hardware = self.shared_loader.resolve_references_in_config(self._hardware)
        self._algorithms = self.shared_loader.resolve_references_in_config(self._algorithms)
        self._workflows = self.shared_loader.resolve_references_in_config(self._workflows)
        self._system = self.shared_loader.resolve_references_in_config(self._system)

    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware domain configuration"""
        return self._hardware

    def get_algorithms_config(self) -> Dict[str, Any]:
        """Get algorithms domain configuration"""
        return self._algorithms

    def get_workflows_config(self) -> Dict[str, Any]:
        """Get workflows domain configuration"""
        return self._workflows

    def get_system_config(self) -> Dict[str, Any]:
        """Get system domain configuration"""
        return self._system

    def get_shared_config(self) -> Dict[str, Any]:
        """Get shared configuration"""
        return self.shared_loader.all_shared

    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration from hardware domain"""
        return self._hardware.get("camera", {})

    def get_gimbal_config(self) -> Dict[str, Any]:
        """Get gimbal configuration from hardware domain"""
        return self._hardware.get("gimbal", {})

    def get_sensors_config(self) -> Dict[str, Any]:
        """Get sensors configuration from hardware domain"""
        return self._hardware.get("sensors", {})

    def get_vision_config(self) -> Dict[str, Any]:
        """Get vision configuration from algorithms domain"""
        return self._algorithms.get("vision", {})

    def get_positioning_config(self) -> Dict[str, Any]:
        """Get positioning configuration from algorithms domain"""
        return self._algorithms.get("positioning", {})

    def get_auto_framing_config(self) -> Dict[str, Any]:
        """Get auto-framing configuration from algorithms domain"""
        return self._algorithms.get("auto_framing", {})

    def get_photo_capture_workflow(self) -> Dict[str, Any]:
        """Get photo capture workflow configuration"""
        return self._workflows.get("photo_capture", {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration from system domain"""
        return self._system.get("performance", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration from system domain"""
        return self._system.get("logging", {})

    def get_ros2_config(self) -> Dict[str, Any]:
        """Get ROS2 configuration from system domain"""
        return self._system.get("ros2", {})

    def validate_all_configs(self, use_pydantic: bool = True, check_consistency: bool = True) -> bool:
        """validate all loaded configurations"""
        # Validate shared configs first
        self.shared_loader.validate_shared_configs()

        # Basic validation - check required configs exist
        required_hardware = ["camera", "gimbal", "sensors"]
        for key in required_hardware:
            if key not in self._hardware:
                raise ValueError(f"Missing required hardware config: {key}")

        required_algorithms = ["vision", "positioning", "auto_framing"]
        for key in required_algorithms:
            if key not in self._algorithms:
                raise ValueError(f"Missing required algorithms config: {key}")

        required_workflows = ["photo_capture"]
        for key in required_workflows:
            if key not in self._workflows:
                raise ValueError(f"Missing required workflow config: {key}")

        required_system = ["performance", "logging", "ros2"]
        for key in required_system:
            if key not in self._system:
                raise ValueError(f"Missing required system config: {key}")

        # production Pydantic validation
        if use_pydantic and VALIDATION_AVAILABLE:
            try:
                # Validate all domains and get typed models
                hw, alg, wf, sys = validate_all_domains(
                    self._hardware,
                    self._algorithms,
                    self._workflows,
                    self._system
                )

                # Cross-domain consistency checks
                if check_consistency:
                    from config.validators import validate_domain_consistency
                    validate_domain_consistency(hw, alg, wf, sys)

                print("Production-grade validation passed")
                print("  Type checking")
                print("  Constraint validation")
                print("  Cross-field validation")
                if check_consistency:
                    print("  Cross-domain consistency")

            except Exception as e:
                raise ValueError(f"Production validation failed: {e}")
        elif use_pydantic and not VALIDATION_AVAILABLE:
            print("Pydantic validation requested but not available")

        return True

    def get_validation_summary(self) -> str:
        """get formatted validation summary"""
        if not VALIDATION_AVAILABLE:
            return "validators not available"

        try:
            from config.validators import get_validation_summary

            # Validate and get typed models
            hw, alg, wf, sys = validate_all_domains(
                self._hardware,
                self._algorithms,
                self._workflows,
                self._system
            )

            return get_validation_summary(hw, alg, wf, sys)
        except Exception as e:
            return f"Could not generate summary: {e}"

    def get_config_summary(self) -> str:
        """Get a summary of all loaded configurations"""
        return f"""
    Domain Configuration Summary:
    =============================

    Hardware Domain:
      Configs loaded: {list(self._hardware.keys())}

    Algorithms Domain:
      Configs loaded: {list(self._algorithms.keys())}
      Auto-framing sub-configs: {list(self._algorithms.get('auto_framing', {}).keys())}

    Workflows Domain:
      Configs loaded: {list(self._workflows.keys())}

    System Domain:
      Configs loaded: {list(self._system.keys())}

    Shared Configurations:
      {self.shared_loader.get_config_summary()}
    """

_domain_loader_instance: Optional[DomainConfigLoader] = None


def get_domain_loader(config_dir: Optional[Path] = None) -> DomainConfigLoader:
    """get or create the domain config loader singleton"""
    global _domain_loader_instance

    if _domain_loader_instance is None:
        _domain_loader_instance = DomainConfigLoader(config_dir)

    return _domain_loader_instance


def reset_domain_loader():
    """reset the domain loader"""
    global _domain_loader_instance
    _domain_loader_instance = None

def get_hardware_config() -> Dict[str, Any]:
    """Get hardware configuration"""
    return get_domain_loader().get_hardware_config()


def get_algorithms_config() -> Dict[str, Any]:
    """Get algorithms configuration"""
    return get_domain_loader().get_algorithms_config()


def get_workflows_config() -> Dict[str, Any]:
    """Get workflows configuration"""
    return get_domain_loader().get_workflows_config()


def get_system_config() -> Dict[str, Any]:
    """Get system configuration"""
    return get_domain_loader().get_system_config()


def get_camera_config() -> Dict[str, Any]:
    """Get camera configuration"""
    return get_domain_loader().get_camera_config()


def get_vision_config() -> Dict[str, Any]:
    """Get vision configuration"""
    return get_domain_loader().get_vision_config()


def get_auto_framing_config() -> Dict[str, Any]:
    """Get auto-framing configuration"""
    return get_domain_loader().get_auto_framing_config()
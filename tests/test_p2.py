

# !/usr/bin/env python3
"""Simple test for Phase 2 components without ROS2"""

import sys
import time

# Test imports
print("Testing Phase 2 imports...")
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.settings import get_settings, get_workflow_config

    print("✓ Settings loaded")

    from mipcs.vision.depth_processor import DepthProcessor

    print("✓ DepthProcessor imported")

    from mipcs.positioning.position_calculator import PositionCalculator
    from mipcs.positioning.scene_classifier import SceneClassifier

    print("✓ Positioning modules imported")

    from mipcs.control.state_machine import PhotoStateMachine

    print("✓ State machine imported")

    from mipcs.utils.voice_guidance import VoiceGuidanceMapper

    print("✓ Voice guidance imported")

    # Test initialization (without ROS2)
    print("\nTesting component initialization...")

    workflow_config = get_workflow_config()
    print(f"✓ Workflow config loaded (language: {workflow_config.voice_guidance.language})")

    voice_mapper = VoiceGuidanceMapper()
    msg = voice_mapper.get_message("welcome")
    print(f"✓ Voice mapper working: '{msg}'")

    state_machine = PhotoStateMachine()
    print(f"✓ State machine initialized (state: {state_machine.current_state.name})")

    print("\n✅ All Phase 2 components working!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

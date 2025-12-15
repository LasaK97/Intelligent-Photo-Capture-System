#!/usr/bin/env python3
"""
Test script to verify configuration loading
Run this from your project root directory
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_main_config():
    """Test loading main config.yaml"""
    print("=" * 60)
    print("Testing main config (config.yaml)")
    print("=" * 60)

    try:
        from config.settings import get_settings
        settings = get_settings()

        print("✓ Main config loaded successfully!")
        print(f"  - System: {settings.system.name} v{settings.system.version}")
        print(f"  - Environment: {settings.system.environment}")
        print(f"  - YOLO model: {settings.vision.yolo.model_path}")
        print(f"  - Target FPS: {settings.performance.target_fps}")
        print(f"  - Gimbal control: {settings.camera_control.gimbal.control_method}")
        print(f"  - Focus topic: {settings.camera_control.focus.control_topic}")
        print()
        return True

    except Exception as e:
        print(f"❌ Error loading main config: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_config():
    """Test loading photo_capture.yaml"""
    print("=" * 60)
    print("Testing workflow config (photo_capture.yaml)")
    print("=" * 60)

    try:
        from config.settings import get_workflow_config
        wc = get_workflow_config()

        print("✓ Workflow config loaded successfully!")
        print(f"  - Language: {wc.voice_guidance.language}")
        print(f"  - Welcome messages: {len(wc.voice_guidance.messages.welcome)}")
        print(f"  - States defined: {len(wc.workflow.states.__fields__)}")
        print(f"  - Error handling enabled: {wc.error_handling.system_failures.emergency_shutdown_enabled}")
        print()

        # Test message retrieval
        print("Sample messages:")
        print(f"  - Welcome: '{wc.voice_guidance.messages.welcome[0]}'")
        print(f"  - Countdown: {wc.voice_guidance.messages.countdown_numbers}")
        print()

        return True

    except Exception as e:
        print(f"❌ Error loading workflow config: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice_guidance_mapper():
    """Test VoiceGuidanceMapper"""
    print("=" * 60)
    print("Testing VoiceGuidanceMapper")
    print("=" * 60)

    try:
        from mipcs.utils.voice_guidance import VoiceGuidanceMapper

        vm = VoiceGuidanceMapper()
        print("✓ VoiceGuidanceMapper initialized!")

        # Test getting messages
        welcome_msg = vm.get_message('welcome')
        countdown_msg = vm.get_message('countdown_start')

        print(f"  - Random welcome: '{welcome_msg}'")
        print(f"  - Random countdown: '{countdown_msg}'")
        print()

        # Test multiple messages
        keys = ['move_closer', 'move_further', 'perfect_position']
        messages = vm.get_messages_for_actions(keys)
        print(f"  - Got {len(messages)} messages for {len(keys)} action keys")
        print()

        return True

    except Exception as e:
        print(f"❌ Error testing VoiceGuidanceMapper: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CONFIGURATION VALIDATION TEST SUITE")
    print("=" * 60)
    print()

    results = {
        "Main Config": test_main_config(),
        "Workflow Config": test_workflow_config(),
        "Voice Guidance Mapper": test_voice_guidance_mapper()
    }

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status:8} - {test_name}")

    all_passed = all(results.values())
    print()
    if all_passed:
        print("🎉 All tests passed! Configuration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
    sys.exit(main())
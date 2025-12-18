"""
Test Shared Configuration Loader
=================================
Tests that shared configs load correctly and reference resolution works
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from shared_loader import (
    SharedConfigLoader,
    get_shared_loader,
    get_camera_specs,
    get_gimbal_specs,
    get_ros_topics,
    resolve_reference
)


def test_shared_loader_initialization():
    """Test that shared loader initializes correctly"""
    print("\n" + "=" * 70)
    print("TEST 1: Shared Loader Initialization")
    print("=" * 70)

    try:
        loader = SharedConfigLoader()
        print("✅ Shared loader initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False


def test_load_all_shared_configs():
    """Test that all shared configs load"""
    print("\n" + "=" * 70)
    print("TEST 2: Load All Shared Configs")
    print("=" * 70)

    try:
        loader = get_shared_loader()

        # Test hardware specs
        hardware = loader.hardware_specs
        print(f"✅ Hardware specs loaded: {len(hardware)} top-level keys")
        print(f"   - Camera mount height: {hardware['camera']['mount_height']}m")
        print(f"   - Gimbal joints: {hardware['gimbal']['joint_names']}")

        # Test coordinate frames
        frames = loader.coordinate_frames
        print(f"✅ Coordinate frames loaded: {len(frames['frames'])} frames")
        print(f"   - Camera optical: {frames['frames']['camera_optical']}")

        # Test ROS topics
        topics = loader.ros_topics
        print(f"✅ ROS topics loaded")
        print(f"   - Input topics: {len(topics['input_topics'])}")
        print(f"   - Output topics: {len(topics['output_topics'])}")

        # Test defaults
        defaults = loader.defaults
        print(f"✅ System defaults loaded")
        print(f"   - System name: {defaults['system']['name']}")
        print(f"   - Version: {defaults['system']['version']}")

        return True
    except Exception as e:
        print(f"❌ Failed to load configs: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_accessors():
    """Test convenience accessor methods"""
    print("\n" + "=" * 70)
    print("TEST 3: Convenience Accessors")
    print("=" * 70)

    try:
        loader = get_shared_loader()

        # Test camera mount height
        height = loader.get_camera_mount_height()
        print(f"✅ Camera mount height: {height}m")
        assert height == 1.68, f"Expected 1.68, got {height}"

        # Test focal length range
        min_focal, max_focal = loader.get_focal_length_range()
        print(f"✅ Focal length range: {min_focal}-{max_focal}mm")
        assert min_focal == 24.0 and max_focal == 105.0

        # Test gimbal limits
        limits = loader.get_gimbal_limits()
        print(f"✅ Gimbal limits: {list(limits.keys())}")
        assert "yaw" in limits and "pitch" in limits

        # Test frame names
        frames = loader.get_frame_names()
        print(f"✅ Frame names: {list(frames.keys())}")
        assert "camera_optical" in frames

        # Test input topics
        input_topics = loader.get_input_topics()
        print(f"✅ Input topics: {len(input_topics)} topics")
        assert "activation_trigger" in input_topics

        return True
    except Exception as e:
        print(f"❌ Convenience accessor failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reference_resolution():
    """Test reference resolution functionality"""
    print("\n" + "=" * 70)
    print("TEST 4: Reference Resolution")
    print("=" * 70)

    try:
        loader = get_shared_loader()

        # Test simple reference
        height = loader.resolve_reference("hardware.camera.mount_height")
        print(f"✅ Resolved 'hardware.camera.mount_height': {height}m")
        assert height == 1.68

        # Test nested reference
        yaw_min = loader.resolve_reference("hardware.gimbal.limits.yaw.min")
        print(f"✅ Resolved 'hardware.gimbal.limits.yaw.min': {yaw_min} rad")
        assert yaw_min == -1.57

        # Test reference in config dict
        test_config = {
            "positioning": {
                "camera_height": "${shared.hardware.camera.mount_height}",
                "some_other_value": 123
            },
            "lens": {
                "min_focal": "${shared.hardware.camera.lens.focal_length.min}"
            }
        }

        resolved = loader.resolve_references_in_config(test_config)
        print(f"✅ Resolved config references:")
        print(f"   - camera_height: {resolved['positioning']['camera_height']}")
        print(f"   - min_focal: {resolved['lens']['min_focal']}")

        assert resolved['positioning']['camera_height'] == 1.68
        assert resolved['lens']['min_focal'] == 24.0
        assert resolved['positioning']['some_other_value'] == 123

        return True
    except Exception as e:
        print(f"❌ Reference resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation():
    """Test config validation"""
    print("\n" + "=" * 70)
    print("TEST 5: Configuration Validation")
    print("=" * 70)

    try:
        loader = get_shared_loader()
        is_valid = loader.validate_shared_configs()
        print("✅ All shared configs validated successfully")
        return is_valid
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def test_module_level_functions():
    """Test module-level convenience functions"""
    print("\n" + "=" * 70)
    print("TEST 6: Module-Level Functions")
    print("=" * 70)

    try:
        # Test get_camera_specs
        camera = get_camera_specs()
        print(f"✅ get_camera_specs(): {camera['sensor']['type']}")

        # Test get_gimbal_specs
        gimbal = get_gimbal_specs()
        print(f"✅ get_gimbal_specs(): {len(gimbal['joint_names'])} joints")

        # Test get_ros_topics
        topics = get_ros_topics()
        print(f"✅ get_ros_topics(): {len(topics['input_topics'])} input topics")

        # Test resolve_reference function
        height = resolve_reference("hardware.camera.mount_height")
        print(f"✅ resolve_reference(): {height}m")

        return True
    except Exception as e:
        print(f"❌ Module-level functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_summary():
    """Test config summary generation"""
    print("\n" + "=" * 70)
    print("TEST 7: Configuration Summary")
    print("=" * 70)

    try:
        loader = get_shared_loader()
        summary = loader.get_config_summary()
        print(summary)
        print("✅ Config summary generated successfully")
        return True
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 70)
    print("SHARED CONFIG LOADER TEST SUITE")
    print("=" * 70)

    tests = [
        ("Initialization", test_shared_loader_initialization),
        ("Load Configs", test_load_all_shared_configs),
        ("Convenience Accessors", test_convenience_accessors),
        ("Reference Resolution", test_reference_resolution),
        ("Validation", test_validation),
        ("Module Functions", test_module_level_functions),
        ("Config Summary", test_config_summary),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Shared config loader is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
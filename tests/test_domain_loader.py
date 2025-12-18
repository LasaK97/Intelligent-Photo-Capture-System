"""
Test Domain Configuration Loader
=================================
Tests that all domain configs load correctly and accessors work
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.domain_loader import (
    DomainConfigLoader,
    get_domain_loader,
    get_hardware_config,
    get_algorithms_config,
    get_camera_config,
    get_vision_config,
    get_auto_framing_config
)


def test_domain_loader_init():
    """Test domain loader initializes correctly"""
    print("\n" + "=" * 70)
    print("TEST 1: Domain Loader Initialization")
    print("=" * 70)

    try:
        loader = DomainConfigLoader()
        print("✅ Domain loader initialized")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_all_domains():
    """Test all domains load correctly"""
    print("\n" + "=" * 70)
    print("TEST 2: Load All Domain Configs")
    print("=" * 70)

    try:
        loader = get_domain_loader()

        # Test hardware
        hardware = loader.get_hardware_config()
        print(f"✅ Hardware: {list(hardware.keys())}")

        # Test algorithms
        algorithms = loader.get_algorithms_config()
        print(f"✅ Algorithms: {list(algorithms.keys())}")

        # Test auto-framing nested configs
        if 'auto_framing' in algorithms:
            af_keys = list(algorithms['auto_framing'].keys())
            print(f"✅ Auto-framing sub-configs: {af_keys}")

        # Test workflows
        workflows = loader.get_workflows_config()
        print(f"✅ Workflows: {list(workflows.keys())}")

        # Test system
        system = loader.get_system_config()
        print(f"✅ System: {list(system.keys())}")

        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_accessors():
    """Test convenience accessor methods"""
    print("\n" + "=" * 70)
    print("TEST 3: Convenience Accessors")
    print("=" * 70)

    try:
        loader = get_domain_loader()

        # Test specific config accessors
        camera = loader.get_camera_config()
        print(f"✅ Camera config: {len(camera)} keys")

        gimbal = loader.get_gimbal_config()
        print(f"✅ Gimbal config: {len(gimbal)} keys")

        vision = loader.get_vision_config()
        print(f"✅ Vision config: {len(vision)} keys")

        positioning = loader.get_positioning_config()
        print(f"✅ Positioning config: {len(positioning)} keys")

        auto_framing = loader.get_auto_framing_config()
        print(f"✅ Auto-framing config: {len(auto_framing)} keys")

        photo_capture = loader.get_photo_capture_workflow()
        print(f"✅ Photo capture workflow: {len(photo_capture)} keys")

        performance = loader.get_performance_config()
        print(f"✅ Performance config: {len(performance)} keys")

        logging = loader.get_logging_config()
        print(f"✅ Logging config: {len(logging)} keys")

        ros2 = loader.get_ros2_config()
        print(f"✅ ROS2 config: {len(ros2)} keys")

        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_structure():
    """Test config structure and values"""
    print("\n" + "=" * 70)
    print("TEST 4: Config Structure & Values")
    print("=" * 70)

    try:
        loader = get_domain_loader()

        # Test camera config structure
        camera = loader.get_camera_config()
        assert 'exposure_control' in camera
        print(f"✅ Camera exposure_control.enabled = {camera['exposure_control']['enabled']}")

        # Test vision config
        vision = loader.get_vision_config()
        assert 'yolo' in vision
        print(f"✅ Vision YOLO confidence = {vision['yolo']['confidence_threshold']}")

        # Test auto-framing nested structure
        auto_framing = loader.get_auto_framing_config()
        assert 'core' in auto_framing
        assert 'composition' in auto_framing
        assert 'exposure' in auto_framing
        print(f"✅ Auto-framing has 3 sub-configs: core, composition, exposure")

        # Test workflow
        workflow = loader.get_photo_capture_workflow()
        assert 'states' in workflow
        print(f"✅ Workflow has {len(workflow['states'])} states")

        # Test system configs
        performance = loader.get_performance_config()
        assert 'control_timing' in performance
        print(f"✅ Performance control_timing.main_loop_hz = {performance['control_timing']['main_loop_hz']}")

        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation():
    """Test config validation"""
    print("\n" + "=" * 70)
    print("TEST 5: Configuration Validation")
    print("=" * 70)

    try:
        loader = get_domain_loader()
        is_valid = loader.validate_all_configs()
        print("✅ All configs validated successfully")
        return is_valid
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def test_module_functions():
    """Test module-level convenience functions"""
    print("\n" + "=" * 70)
    print("TEST 6: Module-Level Functions")
    print("=" * 70)

    try:
        # Test module-level accessors
        hardware = get_hardware_config()
        print(f"✅ get_hardware_config(): {list(hardware.keys())}")

        algorithms = get_algorithms_config()
        print(f"✅ get_algorithms_config(): {list(algorithms.keys())}")

        camera = get_camera_config()
        print(f"✅ get_camera_config(): {len(camera)} keys")

        vision = get_vision_config()
        print(f"✅ get_vision_config(): {len(vision)} keys")

        auto_framing = get_auto_framing_config()
        print(f"✅ get_auto_framing_config(): {len(auto_framing)} keys")

        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_summary():
    """Test config summary"""
    print("\n" + "=" * 70)
    print("TEST 7: Configuration Summary")
    print("=" * 70)

    try:
        loader = get_domain_loader()
        summary = loader.get_config_summary()
        print(summary)
        print("✅ Config summary generated")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("DOMAIN CONFIG LOADER TEST SUITE")
    print("=" * 70)

    tests = [
        ("Initialization", test_domain_loader_init),
        ("Load Domains", test_load_all_domains),
        ("Convenience Accessors", test_convenience_accessors),
        ("Config Structure", test_config_structure),
        ("Validation", test_validation),
        ("Module Functions", test_module_functions),
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
        print("\n🎉 ALL TESTS PASSED! Domain config loader working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
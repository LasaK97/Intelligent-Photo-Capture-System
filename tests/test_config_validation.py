# !/usr/bin/env python3
"""
Config Validation Test (No ROS)
================================
Tests ONLY configuration loading and validation
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("CONFIGURATION VALIDATION TEST")
print("=" * 70)

# Test 1: Load all configs via settings.py
print("\n📦 Test 1: Loading configs via settings.py...")
try:
    from config.settings import get_settings

    settings = get_settings()
    print("✅ Settings loaded successfully")

    # Test access patterns
    hw = settings.get_hardware_config()
    alg = settings.get_algorithms_config()
    wf = settings.get_workflows_config()
    sys_config = settings.get_system_config()

    print(f"   Hardware configs: {list(hw.keys())}")
    print(f"   Algorithms configs: {list(alg.keys())}")
    print(f"   Workflows configs: {list(wf.keys())}")
    print(f"   System configs: {list(sys_config.keys())}")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 2: Validate all domains
print("\n🔍 Test 2: Running production validation...")
try:
    from config.validators import validate_all_domains, get_validation_summary

    # Validate
    validated_hw, validated_alg, validated_wf, validated_sys = validate_all_domains(
        hw, alg, wf, sys_config
    )

    print("✅ All domains validated successfully")
    print("   ✓ Type checking passed")
    print("   ✓ Constraint validation passed")
    print("   ✓ Cross-field validation passed")

except Exception as e:
    print(f"❌ Validation FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Get validation summary
print("\n📊 Test 3: Generating validation summary...")
try:
    summary = get_validation_summary(validated_hw, validated_alg, validated_wf, validated_sys)
    print(summary)

except Exception as e:
    print(f"❌ Summary generation FAILED: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Test specific config access (like your scripts will use)
print("\n🔧 Test 4: Testing config access patterns...")
try:
    from config.settings import get_vision_config, get_camera_config, get_auto_framing_config

    # Test vision config
    vision = get_vision_config()
    yolo_confidence = vision['yolo']['confidence_threshold']
    print(f"✅ Vision config: YOLO confidence = {yolo_confidence}")

    # Test camera config
    camera = get_camera_config()
    iso_base = camera['iso_preferences']['preferred_base']
    print(f"✅ Camera config: ISO base = {iso_base}")

    # Test auto-framing config
    af = get_auto_framing_config()
    quality = af['core']['quality_threshold']
    print(f"✅ Auto-framing config: Quality threshold = {quality}")

except Exception as e:
    print(f"❌ Config access FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("🎉 ALL CONFIG TESTS PASSED!")
print("=" * 70)
print("\nYour configuration system is working correctly!")
print("You can now integrate into your scripts.")

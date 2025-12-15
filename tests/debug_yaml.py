#!/usr/bin/env python3
"""
Debug script to see what's actually in your photo_capture.yaml file
"""

import os
import yaml
from pathlib import Path

# Check the file path
config_path = os.getenv("WORKFLOW_CONFIG_PATH", "config/photo_capture.yaml")
print(f"Looking for config at: {config_path}")
print(f"Absolute path: {Path(config_path).absolute()}")
print()

# Check if file exists
if not Path(config_path).exists():
    print(f"❌ File does not exist at: {config_path}")
    print()
    print("Searching for photo_capture.yaml files...")
    for yaml_file in Path('.').rglob('photo_capture.yaml'):
        print(f"  Found: {yaml_file}")
    exit(1)

print(f"✓ File exists")
print()

# Load and inspect the YAML
with open(config_path, 'r') as f:
    data = yaml.safe_load(f)

print("=" * 60)
print("YAML Structure Analysis")
print("=" * 60)
print()

print(f"Top-level keys: {list(data.keys())}")
print()

# Check voice_guidance structure
if 'voice_guidance' not in data:
    print("❌ 'voice_guidance' key is MISSING from YAML!")
    exit(1)

vg = data['voice_guidance']
print("voice_guidance keys:", list(vg.keys()))
print()

# Check messages structure
if 'messages' not in vg:
    print("❌ 'messages' key is MISSING from voice_guidance!")
    print("   Available keys:", list(vg.keys()))

    if 'message' in vg:
        print("   ⚠️  Found 'message' (singular) instead!")
        print("   You need to rename 'message:' to 'messages:' in your YAML file")
    exit(1)

messages = vg['messages']
print("✓ 'messages' key found")
print(f"Type: {type(messages)}")
print(f"Keys in messages: {list(messages.keys())}")
print()

# Check for required message keys
required_keys = [
    'welcome', 'single_person_detected', 'couple_detected',
    'group_detected', 'move_closer', 'move_further',
    'move_left', 'move_right', 'move_together', 'spread_out',
    'perfect_position', 'countdown_start', 'countdown_numbers',
    'capture_complete', 'timeout_warning'
]

missing_keys = [key for key in required_keys if key not in messages]

if missing_keys:
    print(f"❌ Missing {len(missing_keys)} required message keys:")
    for key in missing_keys:
        print(f"   - {key}")
else:
    print("✓ All 15 required message keys are present")

print()

# Show sample values
print("Sample message values:")
for key in ['welcome', 'countdown_numbers', 'capture_complete']:
    if key in messages:
        value = messages[key]
        if isinstance(value, list):
            print(f"  {key}: [{len(value)} items] {value[0] if value else 'empty'}")
        else:
            print(f"  {key}: {value}")

print()
print("=" * 60)
print("Recommendation:")
print("=" * 60)

if missing_keys or 'messages' not in vg:
    print("Your YAML file needs the 'voice_guidance.messages' section.")
    print("Copy the structure from the uploaded photo_capture.yaml file.")
else:
    print("YAML structure looks correct!")
    print("The issue might be in how Pydantic is parsing it.")
    print()
    print("Try running:")
    print(
        "  python3 -c \"from config.settings import VoiceGuidanceConfig, VoiceGuidanceMessages; import yaml; data = yaml.safe_load(open('config/photo_capture.yaml')); print(data['voice_guidance'])\"")
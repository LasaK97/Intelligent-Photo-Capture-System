import asyncio
import sys
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mipcs.vision.detector import YOLOPoseDetector
from mipcs.vision.face_analyzer import MediaPipeFaceAnalyzer
from mipcs.vision.frame_client import CanonFrameClient
from mipcs.vision.depth_processor import DepthProcessor


async def test_camera_connection():
    """Test Canon camera connection"""
    print("\n" + "=" * 70)
    print("TEST 1: Canon Camera Connection")
    print("=" * 70)

    client = CanonFrameClient()

    try:
        await asyncio.wait_for(client.start(), timeout=5.0)
        print("✓ Canon camera connected")

        # Try to get a frame
        await asyncio.sleep(1.0)

        if client.latest_frame is not None:
            print(f"✓ Receiving frames: {client.latest_frame.shape}")
        else:
            print("⚠ Connected but no frames received yet")

        await client.stop()
        return True

    except asyncio.TimeoutError:
        print("✗ Camera connection timeout")
        print("  Make sure Canon FrameServer is running")
        return False
    except Exception as e:
        print(f"✗ Camera error: {e}")
        return False


async def test_realsense():
    """Test RealSense depth camera"""
    print("\n" + "=" * 70)
    print("TEST 2: RealSense Depth Camera")
    print("=" * 70)

    try:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline.start(config)
        print("✓ RealSense connected")

        # Get a few frames
        for i in range(5):
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                print(f"✓ Frame {i + 1}: depth data received")

        pipeline.stop()
        return True

    except ImportError:
        print("⚠ pyrealsense2 not installed")
        return False
    except Exception as e:
        print(f"✗ RealSense error: {e}")
        print("  Make sure RealSense camera is connected")
        return False


async def test_full_pipeline():
    """Test complete detection pipeline"""
    print("\n" + "=" * 70)
    print("TEST 3: Full Vision Pipeline")
    print("=" * 70)

    print("\n[1/3] Initializing components...")
    detector = YOLOPoseDetector()
    face_analyzer = MediaPipeFaceAnalyzer()

    print("[2/3] Loading models...")
    await detector.load_model()
    await face_analyzer.load_models()
    print("✓ All models loaded")

    print("\n[3/3] Testing with sample frame...")
    # Create test frame
    test_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

    # Run detection
    detections = await detector.detect_async(test_frame)
    print(f"✓ YOLO detection completed: {len(detections)} persons found")

    # Test face analysis on dummy bbox
    if len(detections) > 0:
        bbox = detections[0].bbox
        face_result = await face_analyzer.analyze_face_orientation(test_frame, bbox)
        print(f"✓ Face analysis completed: {face_result}")

    stats = detector.get_performance_stats()
    print(f"\nPerformance:")
    print(f"  - YOLO inference: {stats['average_inference_time_ms']:.2f}ms")
    print(f"  - FPS capability: {stats['fps_capability']:.1f}")

    return True


async def main():
    print("=" * 70)
    print("PHASE 1 - HARDWARE INTEGRATION TEST")
    print("Testing on Jetson Orin")
    print("=" * 70)

    results = {
        'camera': await test_camera_connection(),
        'realsense': await test_realsense(),
        'pipeline': await test_full_pipeline()
    }

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:15} : {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - READY FOR PRODUCTION")
    else:
        print("⚠ SOME TESTS FAILED - CHECK HARDWARE CONNECTIONS")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
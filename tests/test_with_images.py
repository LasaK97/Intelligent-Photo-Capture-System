import asyncio
import sys
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mipcs.vision.detector import YOLOPoseDetector
from mipcs.vision.face_analyzer import MediaPipeFaceAnalyzer


async def main():
    print("=" * 70)
    print("DETECTION TEST - Using Test Images")
    print("=" * 70)

    # Initialize
    print("\n[1/3] Initializing...")
    detector = YOLOPoseDetector()
    face_analyzer = MediaPipeFaceAnalyzer()

    await detector.load_model()
    await face_analyzer.load_models()
    print("✓ Models loaded")

    # Create test images with people-like shapes
    print("\n[2/3] Creating test frames...")
    test_frames = []

    # Frame 1: Person in center
    frame1 = np.random.randint(100, 150, (640, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (250, 150), (390, 500), (180, 150, 120), -1)
    cv2.circle(frame1, (320, 200), 40, (200, 180, 160), -1)
    test_frames.append(("Center person", frame1))

    # Frame 2: Two people
    frame2 = np.random.randint(100, 150, (640, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (100, 150), (220, 500), (180, 150, 120), -1)
    cv2.rectangle(frame2, (420, 150), (540, 500), (180, 150, 120), -1)
    test_frames.append(("Two people", frame2))

    # Frame 3: Empty scene
    frame3 = np.random.randint(100, 150, (640, 640, 3), dtype=np.uint8)
    test_frames.append(("Empty scene", frame3))

    print(f"✓ Created {len(test_frames)} test frames")

    # Process frames
    print("\n[3/3] Processing frames...")
    print("=" * 70)

    for name, frame in test_frames:
        print(f"\nProcessing: {name}")

        detections = await detector.detect_async(frame)
        print(f"  Detections: {len(detections)}")

        if len(detections) > 0:
            for idx, det in enumerate(detections):
                print(f"  Person {idx + 1}:")
                print(f"    - BBox: {det.bbox}")
                print(f"    - Confidence: {det.confidence:.2f}")
                print(f"    - Pose quality: {det.pose_analysis.get('pose_quality', 0):.2f}")

                # Face analysis
                face_result = await face_analyzer.analyze_face_orientation(frame, det.bbox)
                if face_result:
                    print(f"    - Face: {face_result.orientation}, " +
                          f"facing={face_result.facing_camera}")

    # Final stats
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    stats = detector.get_performance_stats()
    print(f"YOLO Detector:")
    print(f"  - Inferences: {stats['inference_count']}")
    print(f"  - Avg time: {stats.get('average_inference_time_ms', 0):.2f}ms")
    print(f"  - FPS capability: {stats.get('fps_capability', 0):.1f}")

    face_stats = face_analyzer.get_performance_stats()
    print(f"\nFace Analyzer:")
    print(f"  - Analyses: {face_stats['analysis_count']}")
    if face_stats['analysis_count'] > 0:
        print(f"  - Avg time: {face_stats.get('average_analysis_time_ms', 0):.2f}ms")

    print("=" * 70)
    print("✓ TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
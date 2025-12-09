import asyncio
import sys
from pathlib import Path
import cv2
import time

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from mipcs.vision.detector import YOLOPoseDetector
from mipcs.vision.face_analyzer import MediaPipeFaceAnalyzer
from mipcs.vision.frame_client import CanonFrameClient


async def main():
    print("=" * 70)
    print("LIVE DETECTION TEST - Processing Real Camera Frames")
    print("=" * 70)

    # Initialize components
    print("\nInitializing...")
    detector = YOLOPoseDetector()
    face_analyzer = MediaPipeFaceAnalyzer()
    camera = CanonFrameClient()

    await detector.load_model()
    await face_analyzer.load_models()
    await camera.start()

    print("✓ All components ready")
    print("\nWaiting for frames from Canon camera...")
    print("(Make sure Canon FrameServer is actively streaming)")

    # Wait for first frame
    for i in range(50):
        await asyncio.sleep(0.1)
        if camera.latest_frame is not None:
            print(f"✓ First frame received after {i * 0.1:.1f}s")
            break
    else:
        print("\n⚠ No frames received from camera")
        print("Canon FrameServer may not be streaming")
        await camera.stop()
        return

    print("\nProcessing 30 frames...")

    frame_count = 0
    detection_count = 0

    for i in range(30):
        await asyncio.sleep(0.1)

        if camera.latest_frame is not None:
            frame = camera.latest_frame.copy()
            frame_count += 1

            # Run detection
            detections = await detector.detect_async(frame)

            if len(detections) > 0:
                detection_count += 1
                print(f"Frame {frame_count}: Found {len(detections)} person(s)")

                # Analyze faces
                for idx, det in enumerate(detections):
                    face_result = await face_analyzer.analyze_face_orientation(
                        frame, det.bbox
                    )
                    if face_result:
                        print(f"  Person {idx + 1}: {face_result.orientation}, " +
                              f"facing_camera={face_result.facing_camera}, " +
                              f"conf={face_result.confidence:.2f}")
            else:
                if frame_count % 10 == 0:
                    print(f"Frame {frame_count}: No detections")

    await camera.stop()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Frames processed: {frame_count}")
    print(f"Frames with detections: {detection_count}")

    stats = detector.get_performance_stats()
    print(f"\nPerformance:")
    print(f"  - Total inferences: {stats['inference_count']}")
    if stats['inference_count'] > 0:
        print(f"  - Avg inference: {stats.get('average_inference_time_ms', 0):.2f}ms")
        print(f"  - FPS capability: {stats.get('fps_capability', 0):.1f}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import time
from pathlib import Path
import sys
import numpy as np
from typing import Optional, List, Dict, Any
import cv2

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import String

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = None
    MultiThreadedExecutor = None
    String = None

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


from config.settings import get_settings
from ..utils.logger import get_logger, init_logging, log_performance
from ..utils.threading_utils import get_thread_manager, project_root
from ..utils.exceptions import ManriixError

from ..vision.detector import YOLOPoseDetector, PersonDetection
from ..vision.face_analyzer import MediaPipeFaceAnalyzer, FaceAnalysis
from ..vision.frame_client import CanonFrameClient, FrameMetadata
from ..vision.depth_processor import DepthProcessor, DepthData

logger = get_logger(__name__)


class VisionPipelineData:
    """Consolidated data from vision pipeline."""

    def __init__(self):
        self.frame: Optional[np.ndarray] = None
        self.frame_metadata: Optional[FrameMetadata] = None
        self.person_detections: List[PersonDetection] = []
        self.face_analyses: List[Optional[FaceAnalysis]] = []
        self.depth_data: List[Optional[DepthData]] = []
        self.processing_time_ms: float = 0.0
        self.timestamp: float = time.time()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'timestamp': self.timestamp,
            'frame_available': self.frame is not None,
            'frame_shape': self.frame.shape if self.frame is not None else None,
            'person_count': len(self.person_detections),
            'faces_analyzed': sum(1 for fa in self.face_analyses if fa is not None),
            'depth_measurements': sum(1 for dd in self.depth_data if dd is not None),
            'processing_time_ms': self.processing_time_ms
        }


class VisionTestNode(Node if ROS_AVAILABLE else object):
    """
    ROS2 node for testing the complete vision pipeline.
    Integrates YOLO11-pose, MediaPipe face analysis, Canon frames, and RealSense depth.
    """

    def __init__(self):
        if ROS_AVAILABLE:
            super().__init__('vision_test_node')

        # Initialize logging
        init_logging()

        self.settings = get_settings()

        # Vision components
        self.yolo_detector: Optional[YOLOPoseDetector] = None
        self.face_analyzer: Optional[MediaPipeFaceAnalyzer] = None
        self.frame_client: Optional[CanonFrameClient] = None
        self.depth_processor: Optional[DepthProcessor] = None

        # State
        self.running = False
        self.components_initialized = False

        # Performance tracking
        self.pipeline_runs = 0
        self.total_pipeline_time = 0.0
        self.latest_results: Optional[VisionPipelineData] = None

        # ROS interfaces (if available)
        if ROS_AVAILABLE:
            self.status_publisher = self.create_publisher(
                String,
                '/vision_test/status',
                10
            )

            # Timer for periodic pipeline execution
            self.pipeline_timer = self.create_timer(
                1.0 / self.settings.performance.target_fps,  # Target FPS
                self._pipeline_timer_callback
            )

        logger.info("vision_test_node_initialized")

    async def initialize_components(self) -> None:
        """Initialize all vision pipeline components."""
        if self.components_initialized:
            logger.warning("components_already_initialized")
            return

        logger.info("initializing_vision_pipeline_components")

        try:
            # Initialize YOLO detector
            logger.info("initializing_yolo_detector")
            self.yolo_detector = YOLOPoseDetector()
            await self.yolo_detector.load_model()

            # Initialize face analyzer
            logger.info("initializing_face_analyzer")
            self.face_analyzer = MediaPipeFaceAnalyzer()
            await self.face_analyzer.load_models()

            # Initialize frame client
            logger.info("initializing_frame_client")
            self.frame_client = CanonFrameClient()
            await self.frame_client.start()

            # Initialize depth processor
            logger.info("initializing_depth_processor")
            self.depth_processor = DepthProcessor(self if ROS_AVAILABLE else None)

            if ROS_AVAILABLE:
                # Wait for camera calibration
                calibration_ready = await self.depth_processor.wait_for_calibration(timeout=15.0)
                if not calibration_ready:
                    logger.warning("camera_calibration_timeout_proceeding_anyway")
            else:
                # Set mock intrinsics for testing without ROS
                self.depth_processor.set_camera_intrinsics(
                    fx=640.0, fy=640.0, cx=640.0, cy=360.0,
                    width=1280, height=720
                )

            self.components_initialized = True
            logger.info("all_vision_components_initialized")

        except Exception as e:
            logger.error("component_initialization_failed", error=str(e))
            raise ManriixError(f"Failed to initialize vision components: {e}") from e

    async def start_pipeline(self) -> None:
        """Start the vision pipeline processing."""
        if self.running:
            logger.warning("pipeline_already_running")
            return

        if not self.components_initialized:
            await self.initialize_components()

        self.running = True
        logger.info("vision_pipeline_started")

        # Validate all components
        await self._validate_all_components()

    async def stop_pipeline(self) -> None:
        """Stop the vision pipeline and cleanup."""
        if not self.running:
            return

        logger.info("stopping_vision_pipeline")
        self.running = False

        # Cleanup components
        if self.frame_client:
            await self.frame_client.stop()

        if self.yolo_detector:
            await self.yolo_detector.cleanup()

        if self.face_analyzer:
            await self.face_analyzer.cleanup()

        logger.info("vision_pipeline_stopped")

    @log_performance("vision_pipeline_execution")
    async def run_vision_pipeline(self) -> Optional[VisionPipelineData]:
        """
        Execute complete vision pipeline on latest frame.

        Returns:
            VisionPipelineData with all processing results
        """
        if not self.components_initialized or not self.running:
            return None

        start_time = time.time()
        pipeline_data = VisionPipelineData()

        try:
            # Get latest frame
            frame_result = await self._get_latest_frame()
            if frame_result is None:
                logger.debug("no_frame_available")
                return None

            pipeline_data.frame, pipeline_data.frame_metadata = frame_result

            # Step 1: YOLO person detection + pose
            person_detections = await self.yolo_detector.detect_async(pipeline_data.frame)
            pipeline_data.person_detections = person_detections

            if not person_detections:
                logger.debug("no_persons_detected")
                pipeline_data.processing_time_ms = (time.time() - start_time) * 1000
                return pipeline_data

            logger.debug("persons_detected", count=len(person_detections))

            # Step 2: Face analysis for detected persons
            face_bboxes = [detection.bbox for detection in person_detections]
            face_analyses = await self.face_analyzer.analyze_multiple_faces(
                pipeline_data.frame, face_bboxes
            )
            pipeline_data.face_analyses = face_analyses

            # Step 3: Depth extraction for detected persons
            depth_results = await self.depth_processor.extract_multiple_depths(face_bboxes)
            pipeline_data.depth_data = depth_results

            # Calculate total processing time
            pipeline_data.processing_time_ms = (time.time() - start_time) * 1000

            # Update performance metrics
            self._update_pipeline_metrics(pipeline_data.processing_time_ms)

            # Store latest results
            self.latest_results = pipeline_data

            # Log pipeline results
            summary = pipeline_data.get_summary()
            logger.info("vision_pipeline_completed", **summary)

            return pipeline_data

        except Exception as e:
            logger.error("vision_pipeline_error", error=str(e))
            return None

    async def _get_latest_frame(self) -> Optional[tuple]:
        """Get latest frame from Canon client."""
        if not self.frame_client:
            return None

        # Try to get frame with short timeout
        frame_result = await self.frame_client.get_latest_frame(timeout=0.1)

        if frame_result is None:
            # Fallback to synchronous latest frame
            return self.frame_client.get_latest_frame_sync()

        return frame_result

    def _pipeline_timer_callback(self) -> None:
        """ROS timer callback for periodic pipeline execution."""
        if not self.running:
            return

        # Create async task for pipeline execution
        asyncio.create_task(self._async_pipeline_callback())

    async def _async_pipeline_callback(self) -> None:
        """Async pipeline execution for ROS timer."""
        try:
            results = await self.run_vision_pipeline()

            if results and ROS_AVAILABLE:
                # Publish status message
                status_msg = String()
                status_msg.data = f"Pipeline: {results.get_summary()}"
                self.status_publisher.publish(status_msg)

        except Exception as e:
            logger.error("async_pipeline_callback_error", error=str(e))

    async def _validate_all_components(self) -> Dict[str, Any]:
        """Validate all pipeline components."""
        validation_results = {}

        # Validate YOLO detector
        if self.yolo_detector:
            try:
                yolo_validation = await self.yolo_detector.validate_model()
                validation_results['yolo_detector'] = yolo_validation
            except Exception as e:
                validation_results['yolo_detector'] = {'error': str(e)}

        # Validate frame client
        if self.frame_client:
            try:
                frame_validation = await self.frame_client.validate_connection()
                validation_results['frame_client'] = frame_validation
            except Exception as e:
                validation_results['frame_client'] = {'error': str(e)}

        # Validate depth processor
        if self.depth_processor:
            try:
                depth_validation = await self.depth_processor.validate_depth_processing()
                validation_results['depth_processor'] = depth_validation
            except Exception as e:
                validation_results['depth_processor'] = {'error': str(e)}

        logger.info("component_validation_completed", **validation_results)
        return validation_results

    def _update_pipeline_metrics(self, processing_time_ms: float) -> None:
        """Update pipeline performance metrics."""
        self.pipeline_runs += 1
        self.total_pipeline_time += processing_time_ms

        # Check performance targets
        target_fps = self.settings.performance.target_fps
        target_time_ms = 1000.0 / target_fps

        if processing_time_ms > target_time_ms:
            logger.warning(
                "pipeline_performance_below_target",
                processing_time_ms=processing_time_ms,
                target_time_ms=target_time_ms,
                target_fps=target_fps
            )

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            'pipeline_runs': self.pipeline_runs,
            'components_initialized': self.components_initialized,
            'running': self.running
        }

        if self.pipeline_runs > 0:
            avg_time = self.total_pipeline_time / self.pipeline_runs
            actual_fps = 1000.0 / avg_time if avg_time > 0 else 0

            stats.update({
                'average_processing_time_ms': round(avg_time, 2),
                'total_processing_time_ms': round(self.total_pipeline_time, 2),
                'actual_fps': round(actual_fps, 1),
                'target_fps': self.settings.performance.target_fps,
                'meets_performance_target': actual_fps >= self.settings.performance.target_fps * 0.8
            })

        # Component-specific stats
        if self.yolo_detector:
            stats['yolo_detector'] = self.yolo_detector.get_performance_stats()

        if self.face_analyzer:
            stats['face_analyzer'] = self.face_analyzer.get_performance_stats()

        if self.frame_client:
            stats['frame_client'] = self.frame_client.get_performance_stats()

        if self.depth_processor:
            stats['depth_processor'] = self.depth_processor.get_performance_stats()

        return stats

    async def run_performance_benchmark(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmark.

        Args:
            duration_seconds: How long to run the benchmark

        Returns:
            Benchmark results
        """
        logger.info("starting_performance_benchmark", duration=duration_seconds)

        if not self.components_initialized:
            await self.initialize_components()

        # Clear performance counters
        initial_runs = self.pipeline_runs
        initial_time = self.total_pipeline_time

        benchmark_start = time.time()
        benchmark_runs = 0
        benchmark_errors = 0

        # Run pipeline for specified duration
        while (time.time() - benchmark_start) < duration_seconds:
            try:
                results = await self.run_vision_pipeline()
                if results:
                    benchmark_runs += 1
                else:
                    benchmark_errors += 1

            except Exception as e:
                logger.warning("benchmark_pipeline_error", error=str(e))
                benchmark_errors += 1

            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)

        benchmark_duration = time.time() - benchmark_start
        benchmark_time = self.total_pipeline_time - initial_time

        benchmark_results = {
            'duration_seconds': round(benchmark_duration, 2),
            'pipeline_runs': benchmark_runs,
            'pipeline_errors': benchmark_errors,
            'success_rate': round((benchmark_runs / (benchmark_runs + benchmark_errors)) * 100, 1) if (
                                                                                                                  benchmark_runs + benchmark_errors) > 0 else 0,
            'average_fps': round(benchmark_runs / benchmark_duration, 1) if benchmark_duration > 0 else 0,
            'average_processing_time_ms': round(benchmark_time / benchmark_runs, 2) if benchmark_runs > 0 else 0,
            'target_fps': self.settings.performance.target_fps,
            'performance_stats': self.get_pipeline_stats()
        }

        logger.info("performance_benchmark_completed", **benchmark_results)
        return benchmark_results


# ROS2 Node entry point
def main(args=None):
    """ROS2 node main entry point."""
    if not ROS_AVAILABLE:
        print("ROS2 not available. Cannot run ROS node.")
        return

    rclpy.init(args=args)

    # Create and configure executor
    executor = MultiThreadedExecutor(num_threads=4)

    try:
        # Create node
        node = VisionTestNode()

        # Add node to executor
        executor.add_node(node)

        # Create async task for component initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def init_and_start():
            await node.start_pipeline()
            return node

        # Initialize components
        node = loop.run_until_complete(init_and_start())

        logger.info("vision_test_node_ready_spinning")

        # Spin the executor
        try:
            executor.spin()
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt_received")
        finally:
            # Cleanup
            loop.run_until_complete(node.stop_pipeline())

    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# Standalone testing function (without ROS)
async def run_standalone_test():
    """Run vision pipeline test without ROS dependencies."""
    logger.info("running_standalone_vision_test")

    try:
        # Create test node
        node = VisionTestNode()

        # Initialize components
        await node.initialize_components()

        # Run benchmark
        benchmark_results = await node.run_performance_benchmark(duration_seconds=10)

        print("\n" + "=" * 50)
        print("VISION PIPELINE BENCHMARK RESULTS")
        print("=" * 50)

        for key, value in benchmark_results.items():
            print(f"{key}: {value}")

        print("=" * 50)

        # Cleanup
        await node.stop_pipeline()

        logger.info("standalone_test_completed")

    except Exception as e:
        logger.error("standalone_test_failed", error=str(e))
        raise


# Entry point for standalone testing
if __name__ == "__main__" and not ROS_AVAILABLE:
    asyncio.run(run_standalone_test())
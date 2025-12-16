import asyncio
import time
import threading
from argparse import Action
from importlib.metadata import metadata
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum, auto
import sys

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

#ros2 messages
from std_msgs.msg import String, Int32, Bool
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from sympy.solvers.ode.riccati import remove_redundant_sols
from torch._prims import executor
from dji_rs3pro_ros_controller.msg import GimbalCmd

from ..utils.voice_guidance import VoiceGuidanceMapper

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ..vision.detector import YOLOPoseDetector, PersonDetection
from ..vision.face_analyzer import MediaPipeFaceAnalyzer, FaceAnalysis
from ..vision.frame_client import CanonFrameClient, FrameMetadata
from ..vision.depth_processor import DepthProcessor, DepthData

from ..positioning.position_calculator import PositionCalculator, PersonPosition
from ..positioning.scene_classifier import SceneClassifier, SceneAnalysis
from ..control.camera_controller import CameraController
from ..control.state_machine import PhotoStateMachine, PhotoState
from ..positioning.transform_manager import TransformManager

from config.settings import get_settings, get_workflow_config
from ..utils.logger import get_logger, log_performance
from ..utils.threading_utils import get_thread_manager, AsyncLock
from ..utils.exceptions import ManriixError, VisionError, ControlError

logger = get_logger(__name__)

@dataclass
class ProcessingResult:
    """Container for compiling results"""
    frame_metadata: Optional[FrameMetadata] = None
    person_detections: List[PersonDetection] = None
    face_analyses: List[Optional[FaceAnalysis]] = None
    person_positions: List[PersonPosition] = None
    scene_analysis: Optional[SceneAnalysis] = None
    processing_time_ms: float = 0.0
    success: bool = False


class PhotoCaptureNode(Node):

    def __init__(self):
        super().__init__('photo_capture_node')

        # load configs
        self.settings = get_settings()
        self.config = self.settings.photo_capture

        # processing state
        self.processing_lock = AsyncLock()
        self.latest_results: Optional[ProcessingResult] = None

        #performance monitoring
        self.processing_stats = {
            'frames_processed': 0,
            'total_processing_time': 0.0,
            'average_fps': 0.0,
            'last_processing_time': 0.0
        }

        #init core components
        self._init_components()

        #setup ros2 interfaces
        self._init_ros_interfaces()

        #start processing threads
        self._start_processing_threads()

        logger.info("photo_capture_node_initialized",
                    node_name=self.get_name(),
                    config_loaded=True,
                    components_ready=True)


    def _init_components(self) -> None:
        """Initialize all components"""
        try:
            logger.info("initializing components")
            self.yolo_detector = YOLOPoseDetector()
            self.face_analyzer = MediaPipeFaceAnalyzer()
            self.frame_client = CanonFrameClient()
            self.depth_processor = DepthProcessor(node=self)

            # init TransformManager
            self.transform_manager = TransformManager(
                node=self,
                cache_duration=self.settings.positioning.transform_cache_duration,
                use_tf=self.settings.positioning.use_dynamic_transforms
            )
            logger.info("transform_manager_initialized")

            try:
                camera_pos = self.transform_manager.get_camera_position(
                    camera_frame=self.settings.positioning.frames.camera_optical,
                    base_frame=self.settings.positioning.frames.base_link
                )
                if camera_pos:
                    logger.info(
                        "camera_position_from_tf",
                        x=f"{camera_pos.x:.4f}",
                        y=f"{camera_pos.y:.4f}",
                        z=f"{camera_pos.z:.4f}"
                    )
            except Exception as e:
                logger.warning("camera_position_lookup_failed", error=str(e))

            self.position_calculator = PositionCalculator(
                depth_processor=self.depth_processor,
                transform_manager=self.transform_manager
            )
            self.scene_classifier = SceneClassifier()
            self.camera_controller = CameraController(
                node=self,
                transform_manager=self.transform_manager
            )
            self.state_machine = PhotoStateMachine()

            self.workflow_config = get_workflow_config()
            self.voice_mapper = VoiceGuidanceMapper()

            self.thread_manager = get_thread_manager()

            # self.test_timer = self.create_timer(2.0, self.run_integration_test)

            logger.info("all_components_initialized")
        except Exception as e:
            logger.error("component_initialization_failed", error=str(e))
            raise ManriixError(f"Failed to initialize components: {e}") from e

    def _init_ros_interfaces(self) -> None:
        """Initialize ROS interface"""

        #callback groups
        self.vision_callback_group = ReentrantCallbackGroup()
        self.control_callback_group = ReentrantCallbackGroup()
        self.state_callback_group = MutuallyExclusiveCallbackGroup()

        #subscribers
        self.activation_sub = self.create_subscription(
            String,
            self.settings.ros2_topics.activation_trigger,
            self.activation_callback,
            10,
            callback_group=self.state_callback_group
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            self.settings.ros2_topics.joint_states,
            self.camera_controller.joint_state_callback,
            10,
            callback_group=self.control_callback_group
        )

        #publishers
        self.gimbal_cmd_pub = self.create_publisher(
            GimbalCmd,           # TODO: Implement Float64MultiArray
            self.settings.ros2_topics.gimbal_commands,
            10
        )

        self.focus_cmd_pub = self.create_publisher(
            Int32,
            self.settings.ros2_topics.focus_commands,
            10
        )

        self.capture_trigger_pub = self.create_publisher(
            String,
            self.settings.ros2_topics.capture_trigger,
            10
        )

        self.tts_pub = self.create_publisher(
            String,
            self.settings.ros2_topics.tts_output,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            self.settings.ros2_topics.system_status,
            10
        )

        self.detection_debug_pub = self.create_publisher(
            String, #JSON output
            self.settings.ros2_topics.detection_debug,
            10
        )

        #timers for control loops
        self.main_timer = self.create_timer(
            1.0 / self.settings.control_timing.main_loop_hz,
            self.main_control_loop,
            callback_group=self.state_callback_group
        )

        logger.info("ros_interfaces_initialized",
                    subscribers=4,
                    publishers=6,
                    timers=1)

    def _start_processing_threads(self) -> None:
        """Start threads in background"""

        #vision processing thread
        self.vision_thread = threading.Thread(
            target=self._vision_processing_loop,
            daemon=True,
            name="VisionProcessing"

        )
        self.vision_thread.start()

        #component initialization thread
        self.init_thread = threading.Thread(
            target=self._async_component_initialization,
            daemon=True,
            name="ComponentInit"
        )

        self.init_thread.start()

        logger.info("processing_threads_started")

    def _async_component_initialization(self) -> None:
        """Initialize async components in background"""
        try:
            #create event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            #init async components
            loop.run_until_complete(self._init_async_components())

        except Exception as e:
            logger.error("async_initialization_failed", error=str(e))
        finally:
            loop.close()

    async def _init_async_components(self) -> None:
        """Initialize components require async setup"""

        try:
            #init YOLO detector
            await self.yolo_detector.load_model()

            #init face analyzer
            await self.face_analyzer.load_models()

            #frame clinet
            await self.frame_client.start()

            #wait for cam calibration
            if hasattr(self.depth_processor, "wait_for_calibration"):
                await self.depth_processor.wait_for_calibration(timeout=15.0)

            logger.info("async_components_ready")

        except Exception as e:
            logger.error("async_component_init_failed", error=str(e))
            raise

    def _vision_processing_loop(self) -> None:
        """Main vision processing loop"""
        vision_rate = self.settings.control_timing.vision_processing_hz
        loop_period = 1.0 / vision_rate

        logger.info("vision_processing_loop started", target_hz=vision_rate)

        while rclpy.ok():
            loop_start = time.time()

            try:
                # get latest frame
                frame_data = self.frame_client.get_latest_frame_sync()

                if frame_data is not None:
                    frame, metadata = frame_data

                    #processing frame
                    processing_result = self._process_frame(frame, metadata)

                    #update lates results
                    with self.processing_lock:
                        self.latest_results = processing_result
                        self._update_performance_stats(processing_result)

                # sleep --> To maintain target rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, loop_period - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error("vision_processing_error", error=str(e))
                time.sleep(0.1)


    @log_performance("frame_processing")
    def _process_frame(self, frame, metadata: FrameMetadata) -> ProcessingResult:
        """Process single frame"""

        start_time = time.time()
        result = ProcessingResult(frame_metadata=metadata)

        try:
            # Person detections
            person_detections = asyncio.run_coroutine_threadsafe(
                self.yolo_detector.detect_async(frame),
                asyncio.get_event_loop()
            ).result(timeout=0.05)  #50ms

            result.person_detections = person_detections

            if not person_detections:
                result.success = True   # NOTE : valid result but no people
                return result

            # Face analysis
            face_bboxes = [det.bbox for det in person_detections]
            face_analyses = asyncio.run_coroutine_threadsafe(
                self.frace_analyzer.analyze_multiple_faces(frame, face_bboxes),
                asyncio.get_event_loop()
            ).result(timeout=0.08)   # 80ms

            result.face_analyses = face_analyses

            # depth processing and 3D positions
            person_positions = asyncio.run_coroutine_threadsafe(
                self.position_calculator.calculate_positions_batch(
                    person_detections, self.depth_processor
                ),
                asyncio.get_event_loop()
            ).result(timeout=0.05) #50ms

            # Scene classification
            if person_positions:
                scene_analysis = self.scene_classifier.analyze_scene(
                    person_positions, face_analyses
                )
                result.scene_analysis = scene_analysis

            result.success = True

        except asyncio.TimeoutError:
            logger.warning("frame_processing_timeout")
            result.success = False
        except Exception as e:
            logger.error("frame_processing_error", error=str(e))
            result.success = False
        finally:
            result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    def activation_callback(self, msg: String) -> None:
        """Callback for activation message"""
        if msg.data == self.config.activation.trigger_message:
            logger.info("activation_received")
            self.state_machine.trigger_activation()
        else:
            logger.debug("invalid_activation_message", received=msg.data)

    async def main_control_loop(self) -> None:
        """Main control loop --> timer callback (5Hz)"""
        try:
            #get latest processing results
            latest_results = None
            async with self.processing_lock:
                if self.latest_results is not None:
                    latest_results = self.latest_results

            #update state machine
            actions = self.state_machine.update(
                processing_results=latest_results,
                current_time=time.time()
            )

            #execute actions
            self._execute_actions(actions)

            #publish status
            self._publish_status()

        except Exception as e:
            logger.error("main_control_loop_error", error=str(e))

    def _execute_actions(self, actions: Dict[str, Any]) -> None:
        """"Execute actions returned by state machine"""
        if not actions:
            return

        #voice output --> for TTS
        if 'speak' in actions and actions['speak']:
            msg = String()
            msg.data = actions['speak']
            self.tts_pub.publish(msg)
            logger.debug("tts_message_sent", message=actions['speak'])

        # camera movement
        if 'gimbal_target' in actions:
            self.camera_controller.move_gimbal_to(
                actions['gimbal_target'],
                move_time=actions.get('gimbal_time', 1.0)
            )

        # focus adjustment
        if 'focus_position' in actions:
            msg = Int32()
            msg.data = actions['focus_position']
            self.focus_cmd_pub.publish(msg)
            logger.debug("focus_position_message_sent", message=actions['focus_position'])

        # photo capture
        if 'capture_photo' in actions and actions['capture_photo']:
            msg = String()
            msg.data = 'y'
            self.capture_trigger_pub.publish(msg)
            logger.info("capture_photo_message_sent")


    def _publish_status(self) -> None:
        """Publish status message"""
        status_data = {
            'state': self.state_machine.current_state.name,
            'time_in_state': self.state_machine.time_in_state(),
            'processing_fps': self.processing_stats.get('average_fps', 0.0),
            'last_processing_time_ms': self.processing_stats.get('last_processing_time', 0.0),
            'components_ready': self._check_components_ready()
        }

        msg = String()
        msg.data = str(status_data)
        self.status_pub.publish(msg)

    def _check_components_ready(self) -> Dict[str, bool]:
        """Check if components are ready"""
        return {
            'yolo_detector': getattr(self.yolo_detector, 'model_loaded', False),
            'face_analyzer': getattr(self.face_analyzer, 'models_loaded', False),
            'frame_client': getattr(self.frame_client, 'running', False),
            'depth_processor': getattr(self.depth_processor, 'intrinsics_received', False),
            'camera_controller': self.camera_controller.is_ready(),
            'transform_manager': hasattr(self, 'transform_manager') and self.transform_manager is not None
        }

    def _update_performance_stats(self, result: ProcessingResult) -> None:
        """Update performance statistics"""
        self.processing_stats['frames_processed'] += 1
        self.processing_stats['total_processing_time'] += result.processing_time_ms
        self.processing_stats['last_processing_time'] = result.processing_time_ms

        #cal rolling avg
        if self.processing_stats['frames_processed'] > 0:
            avg_time_ms = (self.processing_stats['total_processing_time'] / self.processing_stats['frames_processed'])

            if avg_time_ms > 0:
                self.processing_stats['average_fps'] = avg_time_ms

    async def cleanup(self) -> None:
        """clean up all components"""

        logger.info('starting_cleanup')

        try:
            if hasattr(self.frame_client, 'stop'):
                await self.frame_client.stop()

            if hasattr(self, 'transform_manager'):
                self.transform_manager.invalidate_cache()
                logger.debug("transform_manager_cache_cleared")

            if hasattr(self.yolo_detector, 'cleanup'):
                await self.yolo_detector.cleanup()

            if hasattr(self.face_analyzer, 'cleanup'):
                await self.face_analyzer.cleanup()

            if hasattr(self.camera_controller, 'stop'):
                await self.camera_controller.stop()

            logger.info('cleanup_completed')

        except Exception as e:
            logger.error("cleanup_error", error=str(e))

    def destroy_node(self) -> None:
        """"""
        try:
            if hasattr(self, 'cleanup'):
                asyncio.run(self.cleanup())
        except  Exception as e:
            logger.warning("cleanup_during_destroy_node_failed", error=str(e))

        #call parent dir
        super().destroy_node()

def main(args=None):
    """main entry point"""
    rclpy.init(args=args)

    #executor -> multi threaded

    executor = MultiThreadedExecutor(num_threads=6)

    try:
        node = PhotoCaptureNode()

        executor.add_node(node)

        logger.info('photo_capture_node_started', node_name=node.get_name())

        #spin
        try:
            executor.spin()
        except KeyboardInterrupt:
            logger.info("keyboard_interruption_received")

    except Exception as e:
        logger.error("node_startup_failed", error=str(e))
    finally:
        #cleanup
        try:
            if 'node' in locals():
                node.destroy_node()
        except Exception as e:
            logger.warning("node_destroy_failed", error=str(e))

        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
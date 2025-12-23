
import time
import asyncio
from pathlib import Path
import sys
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray, Int32
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from ..utils.logger import get_logger, log_performance
from ..utils.threading_utils import AsyncLock
from ..utils.exceptions import ControlError, TimeoutError
from ..positioning.transform_manager import TransformManager
from ..utils.geometry_utils import Point3D


logger = get_logger(__name__)

@dataclass
class GimbalTarget:
    """Gimbal position and movement params"""
    yaw: float  # radians
    pitch: float  # radians
    roll: float  # radians  --->  set to 0 --> Non Roll
    move_time: float  # seconds to reach target
    priority: str  # 'normal', 'urgent', 'emergency'

@dataclass
class GimbalState:
    """Current Gimbal state"""
    position: Tuple[float, float, float]       # yaw, pitch, roll in radians
    velocity: Tuple[float, float, float]       # rad/s
    is_moving: bool
    last_update: float   #timestamp
    target_reached: bool

class FocusMode(Enum):
    """focus calculation modes"""
    HYPERFOCAL = auto()
    DISTANCE_TABLE = auto()
    MANUAL = auto()

class CameraController:
    """Camera controller with gimbal positioning and focus"""

    def __init__(self, node: Node, transform_manager: Optional[TransformManager] = None):
        self.node = node
        self.settings = get_settings()
        self.config = self.settings.camera_control

        self.transform_manager = transform_manager
        positioning_config = self.settings.positioning
        self.camera_frame = positioning_config.frames.camera_optical
        self.base_frame = positioning_config.frames.base_link

        #state tracking
        self.gimbal_state = GimbalState(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            is_moving=False,
            last_update=0.0,
            target_reached=True
        )

        self.current_focus_position: Optional[int] = None
        self.target_focus_position: Optional[int] = None

        #control interface
        self.control_method = self.config.gimbal.control_method
        self.joint_names = self.config.gimbal.joint_names

        #thread safety
        self.state_lock = AsyncLock()
        self.command_lock = AsyncLock()

        #init ros
        self._init_control_interfaces()

        #focus cal
        self._init_focus_system()

        logger.info("camera_controller_initialized", control_method=self.control_method, joint_names = self.joint_names)

    def _init_control_interfaces(self) -> None:
        """Initialise the gimbal control interface"""

        if self.control_method == 'action':
            #action client for trajectory following
            self.action_client = ActionClient(
                self.node,
                FollowJointTrajectory,
                f'/{self.config.gimbal.controller_name}/follow_joint_trajectory'
            )

            #wait for action server
            self.node.create_timer(0.1, self._check_action_server_ready)

        elif self.control_method == 'direct':
            #direct command publisher
            self.gimbal_cmd_pub = self.node.create_publisher(
                Float64MultiArray,
                f'/{self.config.gimbal.controller_name}/commands',
                10
            )
        else:
            raise ControlError(f"Unknown control method '{self.control_method}'")


    def _check_action_server_ready(self) -> None:
        """Check if action server is ready (timer callback)"""
        if hasattr(self, 'action_client') and self.action_client.wait_for_server(timeout_sec=0):
            logger.info("gimbal_action_server_ready")
            #cancel the timer
            if hasattr(self, '_action_check_timer'):
                self._action_check_timer.cancel()

    def get_camera_position(self) -> Optional[Point3D]:
        """
        Get current camera position in base_link frame.
        dynamic lookup --> Uses TransformManager
        """
        if self.transform_manager:
            camera_pos = self.transform_manager.get_camera_position(
                camera_frame=self.camera_frame,
                base_frame=self.base_frame
            )
            if camera_pos:
                return camera_pos
            else:
                logger.warning("camera_position_not_available_from_tf")

        # Fallback to config values
        fallback_pos = Point3D(0.2005, 0.0, 0.72354)
        logger.debug("using_fallback_camera_position")
        return fallback_pos

    def _init_focus_system(self) -> None:
        """Initialise the focus system"""
        focus_config = self.config.focus

        if focus_config.calculation_method == 'hyperfocal':
            self.focus_mode = FocusMode.HYPERFOCAL
            self.hyperfocal_distance = focus_config.hyperfocal_distance

        elif focus_config.calculation_method == 'calibration_table':
            self.focus_mode = FocusMode.DISTANCE_TABLE   #### IMPLEMENT LATER

            # load config file from table
        else:
            self.focus_mode = FocusMode.MANUAL

        logger.info("focus_system_initialized", mode=self.focus_mode.name)

    def joint_state_callback(self, msg: JointState) -> None:
        """update gimble state from joint_states topic"""

        try:
            current_time = time.time()

            #extract gimbal joint positions
            gimbal_positions = []
            gimbal_velocities = []

            for joint_name in self.joint_names:
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    gimbal_positions.append(msg.position[idx])

                    if msg.velocity and len(msg.velocity) > idx:
                        gimbal_velocities.append(msg.velocity[idx])
                    else:
                        gimbal_velocities.append(0.0)
                else:
                    gimbal_positions.append(0.0)
                    gimbal_velocities.append(0.0)

            #update state
            with self.state_lock:
                self.gimbal_state.position = tuple(gimbal_positions)
                self.gimbal_state.velocity = tuple(gimbal_velocities)
                self.gimbal_state.last_update = current_time

                #check if moving (any joint_velocity > threshold)
                velocity_threshold = 0.01 #rad/s
                self.gimbal_state.is_moving = any(
                    abs(v) > velocity_threshold for v in gimbal_velocities
                )

        except Exception as e:
            logger.error("joint_state_processing_error", error=str(e))

    @log_performance("gimbal_movement")
    async def move_gimbal_to(
            self,
            target: GimbalTarget,
            wait_for_completion: bool = False
    ) -> bool:
        """Move gimbal to target position"""
        async with self.command_lock:
            try:
                validated_target = self._validate_gimbal_target(target)

                if self.control_method == 'action':
                    success = await self._move_via_action(validated_target)
                else:
                    success = await self._move_via_direct(validated_target)

                if success and wait_for_completion:
                    await self._wait_for_movement_completion(
                        validated_target,
                        timeout = target.move_time + 2.0
                    )

                logger.debug("gimbal_move_completed", target_yaw=target.yaw, target_pitch=target.pitch, move_time=target.move_time, success=success)

                return success

            except Exception as e:
                logger.error("gimbal_movement_failed", error=str(e))
                raise ControlError(f"Gimbal movement failed: {e}") from e

    def _validate_gimbal_target(self, target: GimbalTarget) -> GimbalTarget:
        """Validate and clamp target to gimbal limits"""
        limits = self.config.gimbal.limits

        #clamp to joint limits
        yaw = np.clip(target.yaw, limits.yaw_range[0], limits.yaw_range[1])
        pitch = np.clip(target.pitch, limits.pitch_range[0], limits.pitch_range[1])
        roll = np.clip(target.roll, limits.roll_range[0], limits.roll_range[1])

        #log if clamping occurs
        if yaw != target.yaw or pitch != target.pitch or roll != target.roll:
            logger.warning("gimbal_target_clamped", original_yaw=target.yaw, clamped_yaw=yaw, original_pitch=target.pitch, clamped_pitch=pitch)

        return GimbalTarget(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            move_time=target.move_time,
            priority=target.priority
        )

    async def _move_via_action(self, target: GimbalTarget) -> bool:
        """Execute movement with action server"""
        if not hasattr(self, 'action_client'):
            raise ControlError("Action client not initialized")

        #craete trajectory goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        #single point trajectory to target
        point = JointTrajectoryPoint()
        point.positions = [target.yaw, target.pitch, target.roll]
        point.velocities = [0.0, 0.0, 0.0] #stop at target
        point.time_from_start = Duration(seconds=target.move_time).to_msg()

        goal.trajectory.points = [point]

        #seng goal
        logger.debug("sending_gimbal_action_goal", target=target)

        send_goal_future = self.action_client.send_goal_async(goal)

        #wait for acceptance
        try:
            goal_handle = await asyncio.wait_for(send_goal_future, timeout = 2.0)

            if not goal_handle.accepted:
                logger.warning("gimbal_goal_rejected")
                return False

            logger.debug("gimbal_goal_accepted")
            return True
        except asyncio.TimeoutError:
            logger.error("gimbal_goal_timeout")
            return False

    async def _move_via_direct(self, target: GimbalTarget) -> bool:
        """Execute movement with direct commands"""
        if not hasattr(self, 'gimbal_cmd_pub'):
            raise ControlError("Gimbal command publisher not initialized")

        #create command message
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [target.yaw, target.pitch, target.roll]

        #publsih command message
        self.gimbal_cmd_pub.publish(cmd_msg)

        logger.debug("sending_gimbal_command_sent", target=target)
        return True

    async def _wait_for_movement_completion(
            self,
            target: GimbalTarget,
            timeout: float
    ) -> bool:
        """Wait for gimbal to reach target position"""
        start_time = time.time()
        position_tolerance = 0.05   #radians (~3 degres)

        while (time.time() - start_time) < timeout:
            with self.state_lock:
                current_pos = self.gimbal_state.position

                #check if close to the target
                yaw_error = abs(current_pos[0] - target.yaw)
                pitch_error = abs(current_pos[1] - target.pitch)
                roll_error = abs(current_pos[2] - target.roll)

                if yaw_error < position_tolerance and pitch_error < position_tolerance and roll_error < position_tolerance:
                    logger.debug("gimbal_target_reached", time_taken=time.time() - start_time)
                    return True

            await asyncio.sleep(0.1)

        logger.warning("gimbal_movement_timeout", timeout=timeout, current_position=self.gimbal_state.position, target_position=(target.yaw, target.pitch, target.roll))
        return False

    def calculate_focus_position(self, distance: float) -> int:
        """Calculate focus position"""
        if self.focus_mode == FocusMode.HYPERFOCAL:
            return self._calculate_hyperfocal_focus(distance)
        elif self.focus_mode == FocusMode.DISTANCE_TABLE:
            return self._calculate_table_focus(distance)
        else:
            return self.current_focus_position or 2048 #middle

    def _calculate_hyperfocal_focus(self, distance: float) -> int:
        """Calculate hyperfocal focus"""
        # For distances beyond hyperfocal, focus at infinity (motor position 0)
        # For closer distances, focus at the distance

        if distance >= self.hyperfocal_distance:
            focus_position = 0
        else:
            max_close_distance = 2.0
            max_motor_position = 4095

            if distance <=max_close_distance:
                #linear interpolation from close focus to infinity
                ratio = (max_close_distance - distance) / max_close_distance
                focus_position = int(ratio * max_motor_position)
            else:
                # Logarithmic scaling for medium distances
                log_ratio = np.log(distance / max_close_distance) / np.log(self.hyperfocal_distance / max_close_distance)
                focus_position = int((1 - log_ratio) * max_motor_position)

        #clamp to valid range
        focus_position = np.clip(focus_position, 0, 4095)

        logger.debug("hyperfocal_focus_calculated", distance=distance, focus_position=focus_position, hyperfocal_distance=self.hyperfocal_distance)

        return focus_position

    async def _calculate_table_focus(self, distance: float) -> int:
        """Calculate table focus"""
        logger.warning("table_focus_not_implemented_using_hyperfocal")   #### IMPLEMENT LATER
        return self._calculate_hyperfocal_focus(distance)

    async def set_focus_position(self, position: int) -> bool:
        """Set focus position"""

        #validate position
        position = np.clip(position, 0, 4095)

        #store target
        self.target_focus_position = position

        logger.debug("focus_position_requested", position=position)

        return True

    def get_current_state(self) -> Dict:
        """Return current state as dict"""
        with self.state_lock:
            return {
                'gimbal': {
                    'position': self.gimbal_state.position,
                    'velocity': self.gimbal_state.velocity,
                    'is_moving': self.gimbal_state.is_moving,
                    'target_reached': self.gimbal_state.target_reached,
                    'last_update': self.gimbal_state.last_update
                },
                'focus': {
                    'current_position': self.current_focus_position,
                    'target_position': self.target_focus_position,
                    'mode': self.focus_mode.name
                },
                'ready': self.is_ready()
            }

    def is_ready(self) -> bool:
        """Check if the camera controller is ready for operation"""
        time_since_update = time.time() - self.gimbal_state.last_update
        gimbal_ready = time_since_update < 2.0  # 2 second timeout

        # Check action server availability if using actions
        action_ready = True
        if self.control_method == 'action':
            action_ready = hasattr(self, 'action_client') and self.action_client.server_is_ready()

        return gimbal_ready and action_ready

    async def stop(self) -> None:
        """Stop camera controller"""
        # cancel any pending actions
        if self.control_method == 'action' and hasattr(self, 'action_client'):
            self.action_client.destroy()

        logger.info("camera_controller_stopped")

    async def execute_trajectory(self, trajectory: Dict) -> bool:
        """
        Execute motion planner trajectory.

        Args:
            trajectory: Dict with 'waypoints', 'duration', 'smoothness'

        Returns:
            True if successful, False otherwise
        """
        waypoints = trajectory.get('waypoints', [])

        if not waypoints:
            logger.warning("empty_trajectory_received")
            return False

        logger.info(
            f"🎬 Executing trajectory: {len(waypoints)} waypoints, "
            f"duration={trajectory.get('duration', 0):.2f}s"
        )

        try:
            for i, waypoint in enumerate(waypoints):
                # Convert degrees to radians (Manriix uses inverted yaw)
                yaw = -np.radians(waypoint['pan'])  # Note: negative for Manriix
                pitch = np.radians(waypoint['tilt'])
                roll = 0.0  # No roll for Manriix

                # Create gimbal target
                target = GimbalTarget(
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    move_time=0.3,  # Smooth 300ms movements
                    priority='normal'
                )

                # Execute waypoint
                success = await self.move_gimbal_to(target)

                if not success:
                    logger.warning(f"⚠️ Waypoint {i + 1}/{len(waypoints)} failed")
                    return False

                logger.debug(f"✅ Waypoint {i + 1}/{len(waypoints)} reached")

                # Small delay between waypoints for smoothness
                if i < len(waypoints) - 1:
                    await asyncio.sleep(0.05)

            logger.info("✅ Trajectory execution complete")
            return True

        except Exception as e:
            logger.error(f"trajectory_execution_error: {e}")
            return False






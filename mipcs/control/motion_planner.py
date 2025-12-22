import numpy as np
import asyncio
import time
import math
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

class MotionProfile(Enum):
    """Motion profile types"""
    LINEAR = "linear"  # Constant velocity
    EASE_IN_OUT = "ease_in_out"  # Smooth acceleration/deceleration
    S_CURVE = "s_curve"  # Smoothest (jerk-limited)
    CUSTOM = "custom"  # User-defined


@dataclass
class Waypoint:
    """Single trajectory waypoint"""
    pan: float  # degrees
    tilt: float  # degrees
    roll: float  # degrees
    zoom: float = 50.0  # mm focal length
    timestamp: float = 0.0  # seconds from start
    velocity: Dict[str, float] = field(default_factory=dict)  # deg/s per axis
    acceleration: Dict[str, float] = field(default_factory=dict)  # deg/s² per axis


@dataclass
class Trajectory:
    """Complete camera trajectory"""
    waypoints: List[Waypoint]
    total_duration: float  # seconds
    max_velocity: Dict[str, float]  # deg/s per axis
    max_acceleration: Dict[str, float]  # deg/s² per axis
    smoothness_score: float  # 0-1 (higher = smoother)
    profile_used: MotionProfile
    safe: bool  # Within all constraints
    violations: List[str] = field(default_factory=list)

class MotionPlanner:

    def __init__(self):
        settings = get_settings()

        # motion constraints
        self.constraints = settings.gimbal.motion_constraints
        self.trajectory_config = settings.gimbal.trajectory

        # velocity limits (deg/s)
        self.max_pan_velocity = self.constraints.max_pan_velocity
        self.max_tilt_velocity = self.constraints.max_tilt_velocity
        self.max_roll_velocity = self.constraints.get('max_roll_velocity', 20.0)

        # acceleration limits (deg/s²)
        self.max_pan_acceleration = self.constraints.max_pan_acceleration
        self.max_tilt_acceleration = self.constraints.max_tilt_acceleration
        self.max_roll_acceleration = self.constraints.get('max_roll_acceleration', 10.0)

        # jerk limit (deg/s³) - rate of change of acceleration
        self.max_jerk = self.constraints.max_jerk

        # smoothness factor (0-1, higher = smoother but slower)
        self.smoothness = self.constraints.smoothness_factor

        # trajectory parameters
        self.sample_rate = self.trajectory_config.default_sample_rate  # Hz
        self.min_points = self.trajectory_config.min_trajectory_points
        self.max_points = self.trajectory_config.max_trajectory_points

        # safety bounds
        self.pan_limits = (-180, 180)
        self.tilt_limits = (-30, 45)
        self.roll_limits = (-20, 20)

        # performance tracking
        self.planning_times = []

        logger.info(
            "motion_planner_initialized",
            max_velocity=f"pan:{self.max_pan_velocity}, tilt:{self.max_tilt_velocity}",
            max_acceleration=f"pan:{self.max_pan_acceleration}, tilt:{self.max_tilt_acceleration}",
            smoothness=self.smoothness
        )

    async def plan_trajectory(
            self,
            start_pose: Dict[str, float],
            target_pose: Dict[str, float],
            duration: Optional[float] = None,
            profile: MotionProfile = MotionProfile.EASE_IN_OUT
    ) -> Trajectory:
        """plan smooth trajectory from start to target."""

        start_time = time.time()

        # extract poses
        start_pan = start_pose.get('pan', 0.0)
        start_tilt = start_pose.get('tilt', 0.0)
        start_roll = start_pose.get('roll', 0.0)
        start_zoom = start_pose.get('zoom', 50.0)

        target_pan = target_pose.get('pan', 0.0)
        target_tilt = target_pose.get('tilt', 0.0)
        target_roll = target_pose.get('roll', 0.0)
        target_zoom = target_pose.get('zoom', 50.0)

        # calculate deltas
        pan_delta = target_pan - start_pan
        tilt_delta = target_tilt - start_tilt
        roll_delta = target_roll - start_roll
        zoom_delta = target_zoom - start_zoom

        # auto-calculate duration if not provided
        if duration is None:
            duration = self._calculate_min_duration(pan_delta, tilt_delta, roll_delta)

        # Check if duration is sufficient for constraints
        min_duration = self._calculate_min_duration(pan_delta, tilt_delta, roll_delta)
        if duration < min_duration:
            logger.warning(
                f"duration_too_short",
                requested=duration,
                minimum=min_duration
            )
            duration = min_duration

        # Generate waypoints based on profile
        if profile == MotionProfile.LINEAR:
            waypoints = self._generate_linear(
                start_pan, start_tilt, start_roll, start_zoom,
                target_pan, target_tilt, target_roll, target_zoom,
                duration
            )
        elif profile == MotionProfile.EASE_IN_OUT:
            waypoints = self._generate_ease_in_out(
                start_pan, start_tilt, start_roll, start_zoom,
                target_pan, target_tilt, target_roll, target_zoom,
                duration
            )
        elif profile == MotionProfile.S_CURVE:
            waypoints = self._generate_s_curve(
                start_pan, start_tilt, start_roll, start_zoom,
                target_pan, target_tilt, target_roll, target_zoom,
                duration
            )
        else:
            waypoints = self._generate_ease_in_out(
                start_pan, start_tilt, start_roll, start_zoom,
                target_pan, target_tilt, target_roll, target_zoom,
                duration
            )

        # Calculate velocities and accelerations
        waypoints = self._calculate_derivatives(waypoints)

        # Check constraints
        safe, violations = self._check_constraints(waypoints)

        # Calculate smoothness
        smoothness = self._calculate_smoothness(waypoints)

        # Get max values
        max_vel, max_accel = self._get_max_values(waypoints)

        # Build trajectory
        trajectory = Trajectory(
            waypoints=waypoints,
            total_duration=duration,
            max_velocity=max_vel,
            max_acceleration=max_accel,
            smoothness_score=smoothness,
            profile_used=profile,
            safe=safe,
            violations=violations
        )

        # Track performance
        elapsed = time.time() - start_time
        self.planning_times.append(elapsed)

        logger.debug(
            "trajectory_planned",
            profile=profile.value,
            waypoints=len(waypoints),
            duration=f"{duration:.2f}s",
            smoothness=f"{smoothness:.2f}",
            safe=safe,
            time_ms=f"{elapsed * 1000:.1f}"
        )

        return trajectory

    def _calculate_min_duration(
            self,
            pan_delta: float,
            tilt_delta: float,
            roll_delta: float
    ) -> float:
        """calculate minimum duration based on velocity limits"""

        # time required for each axis
        pan_time = abs(pan_delta) / self.max_pan_velocity
        tilt_time = abs(tilt_delta) / self.max_tilt_velocity
        roll_time = abs(roll_delta) / self.max_roll_velocity

        # use maximum (bottleneck axis)
        min_duration = max(pan_time, tilt_time, roll_time)

        # add margin for acceleration/deceleration
        min_duration *= 1.5

        # minimum 0.5 seconds
        return max(0.5, min_duration)

    def _generate_linear(
            self,
            start_pan: float, start_tilt: float, start_roll: float, start_zoom: float,
            target_pan: float, target_tilt: float, target_roll: float, target_zoom: float,
            duration: float
    ) -> List[Waypoint]:
        """generate linear motion (constant velocity)"""

        num_points = int(duration * self.sample_rate)
        num_points = np.clip(num_points, self.min_points, self.max_points)

        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)
            timestamp = t * duration

            waypoints.append(Waypoint(
                pan=start_pan + (target_pan - start_pan) * t,
                tilt=start_tilt + (target_tilt - start_tilt) * t,
                roll=start_roll + (target_roll - start_roll) * t,
                zoom=start_zoom + (target_zoom - start_zoom) * t,
                timestamp=timestamp
            ))

        return waypoints

    def _generate_ease_in_out(
            self,
            start_pan: float, start_tilt: float, start_roll: float, start_zoom: float,
            target_pan: float, target_tilt: float, target_roll: float, target_zoom: float,
            duration: float
    ) -> List[Waypoint]:
        """generate ease-in-out motion (cubic easing)"""

        num_points = int(duration * self.sample_rate)
        num_points = np.clip(num_points, self.min_points, self.max_points)

        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)

            # cubic ease-in-out
            if t < 0.5:
                smooth_t = 4 * t * t * t
            else:
                p = 2 * t - 2
                smooth_t = 0.5 * p * p * p + 1

            # apply smoothness factor
            smooth_t = t * (1 - self.smoothness) + smooth_t * self.smoothness

            timestamp = t * duration

            waypoints.append(Waypoint(
                pan=start_pan + (target_pan - start_pan) * smooth_t,
                tilt=start_tilt + (target_tilt - start_tilt) * smooth_t,
                roll=start_roll + (target_roll - start_roll) * smooth_t,
                zoom=start_zoom + (target_zoom - start_zoom) * smooth_t,
                timestamp=timestamp
            ))

        return waypoints

    def _generate_s_curve(
            self,
            start_pan: float, start_tilt: float, start_roll: float, start_zoom: float,
            target_pan: float, target_tilt: float, target_roll: float, target_zoom: float,
            duration: float
    ) -> List[Waypoint]:
        """generate S-curve motion (jerk-limited)"""

        num_points = int(duration * self.sample_rate)
        num_points = np.clip(num_points, self.min_points, self.max_points)

        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)

            # s-curve (quintic polynomial)
            smooth_t = 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

            timestamp = t * duration

            waypoints.append(Waypoint(
                pan=start_pan + (target_pan - start_pan) * smooth_t,
                tilt=start_tilt + (target_tilt - start_tilt) * smooth_t,
                roll=start_roll + (target_roll - start_roll) * smooth_t,
                zoom=start_zoom + (target_zoom - start_zoom) * smooth_t,
                timestamp=timestamp
            ))

        return

    def _calculate_derivatives(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """calculate velocity and acceleration for each waypoint"""

        if len(waypoints) < 2:
            return waypoints

        for i in range(len(waypoints)):
            if i == 0:
                # first point - forward difference
                dt = waypoints[i + 1].timestamp - waypoints[i].timestamp
                if dt > 0:
                    waypoints[i].velocity = {
                        'pan': (waypoints[i + 1].pan - waypoints[i].pan) / dt,
                        'tilt': (waypoints[i + 1].tilt - waypoints[i].tilt) / dt,
                        'roll': (waypoints[i + 1].roll - waypoints[i].roll) / dt
                    }
                else:
                    waypoints[i].velocity = {'pan': 0, 'tilt': 0, 'roll': 0}
                waypoints[i].acceleration = {'pan': 0, 'tilt': 0, 'roll': 0}

            elif i == len(waypoints) - 1:
                # last point - backward difference
                dt = waypoints[i].timestamp - waypoints[i - 1].timestamp
                if dt > 0:
                    waypoints[i].velocity = {
                        'pan': (waypoints[i].pan - waypoints[i - 1].pan) / dt,
                        'tilt': (waypoints[i].tilt - waypoints[i - 1].tilt) / dt,
                        'roll': (waypoints[i].roll - waypoints[i - 1].roll) / dt
                    }
                else:
                    waypoints[i].velocity = {'pan': 0, 'tilt': 0, 'roll': 0}
                waypoints[i].acceleration = {'pan': 0, 'tilt': 0, 'roll': 0}

            else:
                # middle points - central difference
                dt = waypoints[i + 1].timestamp - waypoints[i - 1].timestamp
                if dt > 0:
                    waypoints[i].velocity = {
                        'pan': (waypoints[i + 1].pan - waypoints[i - 1].pan) / dt,
                        'tilt': (waypoints[i + 1].tilt - waypoints[i - 1].tilt) / dt,
                        'roll': (waypoints[i + 1].roll - waypoints[i - 1].roll) / dt
                    }
                else:
                    waypoints[i].velocity = {'pan': 0, 'tilt': 0, 'roll': 0}

                # acceleration from velocity difference
                dt1 = waypoints[i].timestamp - waypoints[i - 1].timestamp
                if dt1 > 0:
                    v_prev = waypoints[i - 1].velocity
                    v_curr = waypoints[i].velocity
                    waypoints[i].acceleration = {
                        'pan': (v_curr['pan'] - v_prev['pan']) / dt1,
                        'tilt': (v_curr['tilt'] - v_prev['tilt']) / dt1,
                        'roll': (v_curr['roll'] - v_prev['roll']) / dt1
                    }
                else:
                    waypoints[i].acceleration = {'pan': 0, 'tilt': 0, 'roll': 0}

        return waypoints

    def _check_constraints(self, waypoints: List[Waypoint]) -> Tuple[bool, List[str]]:
        """check if trajectory violates constraints"""

        violations = []

        for i, wp in enumerate(waypoints):
            # check position limits
            if not (self.pan_limits[0] <= wp.pan <= self.pan_limits[1]):
                violations.append(f"Pan out of bounds at waypoint {i}: {wp.pan:.1f}°")
            if not (self.tilt_limits[0] <= wp.tilt <= self.tilt_limits[1]):
                violations.append(f"Tilt out of bounds at waypoint {i}: {wp.tilt:.1f}°")
            if not (self.roll_limits[0] <= wp.roll <= self.roll_limits[1]):
                violations.append(f"Roll out of bounds at waypoint {i}: {wp.roll:.1f}°")

            # check velocity limits
            if abs(wp.velocity.get('pan', 0)) > self.max_pan_velocity:
                violations.append(f"Pan velocity exceeded at waypoint {i}: {wp.velocity['pan']:.1f}°/s")
            if abs(wp.velocity.get('tilt', 0)) > self.max_tilt_velocity:
                violations.append(f"Tilt velocity exceeded at waypoint {i}: {wp.velocity['tilt']:.1f}°/s")
            if abs(wp.velocity.get('roll', 0)) > self.max_roll_velocity:
                violations.append(f"Roll velocity exceeded at waypoint {i}: {wp.velocity['roll']:.1f}°/s")

            # check acceleration limits
            if abs(wp.acceleration.get('pan', 0)) > self.max_pan_acceleration:
                violations.append(f"Pan acceleration exceeded at waypoint {i}")
            if abs(wp.acceleration.get('tilt', 0)) > self.max_tilt_acceleration:
                violations.append(f"Tilt acceleration exceeded at waypoint {i}")

        safe = len(violations) == 0
        return safe, violations

    def _calculate_smoothness(self, waypoints: List[Waypoint]) -> float:
        """
        Calculate trajectory smoothness (0-1).
        Based on jerk  == rate of change of acceleration.
        """
        if len(waypoints) < 3:
            return 1.0

        jerk_values = []

        for i in range(1, len(waypoints) - 1):
            dt = waypoints[i].timestamp - waypoints[i - 1].timestamp
            if dt > 0:
                # calculate jerk for each axis
                for axis in ['pan', 'tilt', 'roll']:
                    a_prev = waypoints[i - 1].acceleration.get(axis, 0)
                    a_curr = waypoints[i].acceleration.get(axis, 0)
                    jerk = abs((a_curr - a_prev) / dt)
                    jerk_values.append(jerk)

        if not jerk_values:
            return 1.0

        # smoothness = inverse of average jerk (normalized)
        avg_jerk = np.mean(jerk_values)
        smoothness = 1.0 - min(1.0, avg_jerk / self.max_jerk)

        return float(smoothness)

    def _get_max_values(self, waypoints: List[Waypoint]) -> Tuple[Dict, Dict]:
        """get maximum velocity and acceleration values"""

        max_vel = {
            'pan': max(abs(wp.velocity.get('pan', 0)) for wp in waypoints),
            'tilt': max(abs(wp.velocity.get('tilt', 0)) for wp in waypoints),
            'roll': max(abs(wp.velocity.get('roll', 0)) for wp in waypoints)
        }

        max_accel = {
            'pan': max(abs(wp.acceleration.get('pan', 0)) for wp in waypoints),
            'tilt': max(abs(wp.acceleration.get('tilt', 0)) for wp in waypoints),
            'roll': max(abs(wp.acceleration.get('roll', 0)) for wp in waypoints)
        }

        return max_vel, max_accel

    def get_performance_stats(self) -> Dict:
        """get performance statistics"""
        if not self.planning_times:
            return {}

        return {
            'avg_time_ms': np.mean(self.planning_times) * 1000,
            'max_time_ms': np.max(self.planning_times) * 1000,
            'total_trajectories': len(self.planning_times)
        }

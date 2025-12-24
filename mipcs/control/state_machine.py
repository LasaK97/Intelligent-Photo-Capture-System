import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ..positioning.position_calculator import PersonPosition
from ..positioning.scene_classifier import SceneAnalysis, CompositionQuality, SceneType
from config.settings import get_settings
from ..utils.logger import get_logger, log_performance
from ..utils.exceptions import StateMachineError
from ..utils.voice_guidance import VoiceGuidanceMapper

logger = get_logger(__name__)

class PhotoState(Enum):
    """photo capture workflow states"""
    IDLE = auto()  # Waiting for activation
    INITIALIZING = auto()  # Setting up gimbal and focus
    DETECTING = auto()  # Looking for people
    POSITIONING = auto()  # Guiding people to optimal positions
    ADJUSTING_CAMERA = auto()  # Setting optimal gimbal and focus
    VERIFYING = auto()  # Final checks before capture
    COUNTDOWN = auto()  # 3-2-1 countdown
    CAPTURING = auto()  # Taking photo
    COMPLETE = auto()  # Session finished
    ERROR = auto()  # Error state
    TIMEOUT = auto()  # Timeout occurred

@dataclass
class StateContext:
    """context data for the state machine"""
    processing_results: Optional[Any] = None  # latest vision processing results
    person_positions: List[PersonPosition] = None
    scene_analysis: Optional[SceneAnalysis] = None
    current_time: float = 0.0
    session_start_time: float = 0.0

    # state-specific data
    stable_detection_start: Optional[float] = None
    guidance_attempts: int = 0
    last_guidance_time: float = 0.0
    photos_captured: int = 0

    # error tracking
    consecutive_failures: int = 0
    last_error: Optional[str] = None

@dataclass
class StateAction:
    """actions data for the state machine"""
    speak: Optional[str] = None  # TTS message
    gimbal_target: Optional[tuple] = None  # (yaw, pitch, roll)
    gimbal_time: float = 1.0  # Movement time
    focus_position: Optional[int] = None  # Focus motor position
    capture_photo: bool = False  # Trigger photo capture
    transition_to: Optional[PhotoState] = None  # State transition
    delay_seconds: float = 0.0  # Delay before next action

class PhotoStateMachine:
    """Photo capture workflow state machine"""

    def __init__(self):
        self.settings = get_settings()
        self.config = self.settings.photo_capture

        #state management
        self.current_state = PhotoState.IDLE
        self.previous_state: Optional[PhotoState] = None
        self.state_entry_time = time.time()
        self.context = StateContext()

        #state handlers
        self.state_handlers = {
            PhotoState.IDLE: self._handle_idle,
            PhotoState.INITIALIZING: self._handle_initializing,
            PhotoState.DETECTING: self._handle_detecting,
            PhotoState.POSITIONING: self._handle_positioning,
            PhotoState.ADJUSTING_CAMERA: self._handle_adjusting_camera,
            PhotoState.VERIFYING: self._handle_verifying,
            PhotoState.COUNTDOWN: self._handle_countdown,
            PhotoState.CAPTURING: self._handle_capturing,
            PhotoState.COMPLETE: self._handle_complete,
            PhotoState.ERROR: self._handle_error,
            PhotoState.TIMEOUT: self._handle_timeout
        }

        #state timeouts (s)
        self.state_timeouts = {
            PhotoState.INITIALIZING: 10.0,
            PhotoState.DETECTING: 15.0,
            PhotoState.POSITIONING: self.config.workflow.max_positioning_time_s,
            PhotoState.ADJUSTING_CAMERA: 8.0,
            PhotoState.VERIFYING: 5.0,
            PhotoState.COUNTDOWN: self.config.workflow.capture_countdown_s + 1.0,
            PhotoState.CAPTURING: 10.0,
            PhotoState.COMPLETE: 5.0,
            PhotoState.ERROR: 5.0,
            PhotoState.TIMEOUT: 3.0
        }

        #performance tracking
        self.session_stats = {
            'sessions_completed': 0,
            'total_session_time': 0.0,
            'average_session_time': 0.0,
            'state_transitions': 0,
            'errors_encountered': 0
        }

        # voice guidance mapper
        self.voice_mapper = VoiceGuidanceMapper()

        logger.info("photo_state_machine_initialized", initial_state=self.current_state.name)

    def trigger_activation(self) -> None:
        """trigger activation from external source"""
        if self.current_state == PhotoState.IDLE:
            self._transition_to(PhotoState.INITIALIZING, "external_activation")
            self.context.session_start_time = time.time()
        else:
            logger.warning("activation_ignored_not_idle", current_state=self.current_state.name)

    @log_performance("state_machine_update")
    def update(
            self,
            processing_results: Optional[Any] = None,
            current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """update the state machine and returns actions"""
        if current_time is None:
            current_time = time.time()

        #update context
        self.context.processing_results = processing_results
        self.context.current_time = current_time

        #extract relevant data from processing results
        if processing_results and processing_results.success:
            self.context.person_positions = processing_results.person_positions or []

            # if self.context.person_positions:
            #     logger.debug(f"STATE_MACHINE_RECEIVED: {len(self.context.person_positions)} person_positions")

            # TODO:  Implement scene analysis from person positions


        try:
            #check for timeouts
            if self._check_timeout():
                return self._create_action_dict()

                # Execute current state handler
            handler = self.state_handlers.get(self.current_state)
            if handler:
                actions = handler()
                return self._create_action_dict(actions)
            else:
                logger.error("no_handler_for_state", state=self.current_state.name)
                self._transition_to(PhotoState.ERROR, "no_state_handler")
                return self._create_action_dict()

        except Exception as e:
            logger.error("state_machine_update_error", state=self.current_state.name, error=str(e))
            self.context.last_error = str(e)
            self.context.consecutive_failures += 1
            self._transition_to(PhotoState.ERROR, f"state_update_error: {e}")
            return self._create_action_dict()

    def _handle_idle(self) -> Optional[StateAction]:
        """handle idle state -- waiting for activation"""
        return None

    def _handle_initializing(self) -> Optional[StateAction]:
        """handle initializing state -- setup gimbal and focus"""

        time_in_state = self.time_in_state()

        if time_in_state < 0.5:
            #initial setup
            return StateAction(
                speak=self.voice_mapper.get_message("welcome"),
                gimbal_target=(0.0, 0.0, 0.0),
                gimbal_time=2.0,
                focus_position=0  # Start at wide angle (0-4095 range)

            )
        elif time_in_state > 3.0:
            #init complete
            self._transition_to(PhotoState.DETECTING, "initialization_complete")
            return StateAction(
                speak=self.voice_mapper.get_message("welcome")
            )
        return None


    def _handle_detecting(self) -> Optional[StateAction]:
        """handle detecting state -- looking for people"""

        time_in_state = self.time_in_state()
        person_count = len(self.context.person_positions or [])

        if person_count > 0:
            #people detected

            if self.context.stable_detection_start is None:
                self.context.stable_detection_start = time.time()

            #check stable detection for 1 sec
            stable_time = time.time() - self.context.stable_detection_start
            if stable_time > 1.0:
                #classify scene type
                scene_type = self._classify_basic_scene(self.context.person_positions)

                self._transition_to(PhotoState.POSITIONING, f"detected_{person_count}_people")

                if scene_type == SceneType.PORTRAIT:
                    message = self.voice_mapper.get_message("single_person_detected")
                elif scene_type == SceneType.COUPLE:
                    message = self.voice_mapper.get_message("couple_detected")
                elif person_count <= 4:
                    message = f"Wonderful! I can see your group of {person_count}"
                else:
                    message = self.voice_mapper.get_message("group_detected")

                return StateAction(speak=message)
        else:
            #no people detected
            self.context.stable_detection_start = None

            #ecourange people to come in front
            if time_in_state > 5.0 and int(time_in_state) % 3 == 0:
                return StateAction(speak=self.voice_mapper.get_message("welcome"))

        return None

    def _handle_positioning(self) -> Optional[StateAction]:
        """handle positioning state -- guiding people to optimal positions"""

        time_in_state = self.time_in_state()
        current_time = time.time()

        #check if we still have people
        person_count = len(self.context.person_positions or [])
        if person_count == 0:
            self._transition_to(PhotoState.DETECTING, "people_disappeared")
            return StateAction(speak=self.voice_mapper.get_message("welcome"))

        #basic scene analysis
        scene_analysis = self._analyze_basic_scene(self.context.person_positions)

        #check positioning
        if scene_analysis.composition_quality in [CompositionQuality.EXCELLENT, CompositionQuality.GOOD, CompositionQuality.FAIR, CompositionQuality.POOR, CompositionQuality.UNACCEPTABLE]:
            self._transition_to(PhotoState.ADJUSTING_CAMERA, "good_composition_achieved")
            return StateAction(speak=self.voice_mapper.get_message("perfect_position"))

        #guide
        time_since_guidance = current_time - self.context.last_guidance_time
        if time_since_guidance > 3.0:  # Give guidance every 3 seconds

            action_keys = scene_analysis.recommended_action_keys
            if action_keys:
                messages = self.voice_mapper.get_messages_for_actions(action_keys)
                if messages:
                    self.context.last_guidance_time = current_time
                    self.context.guidance_attempts += 1
                    return StateAction(speak=messages[0])

        return None

    def _handle_adjusting_camera(self) -> Optional[StateAction]:
        """handle adjusting camera position --> set optimal gimbal and focus"""

        time_in_state = self.time_in_state()

        if time_in_state < 0.5:
            # calculate optimal camera settings

            if self.context.person_positions:
                scene_analysis = self._analyze_basic_scene(self.context.person_positions)

                #calculate gimbal adjustments
                centroid = scene_analysis.group_centroid
                optimal_distance = scene_analysis.optimal_distance

                #basic gimbal positioning
                yaw_adjustment = np.arctan2(centroid[0], centroid[1])
                pitch_adjustment = 0.0  # TODO: For now level --> check working

                #basic focus calculation
                avg_distance = scene_analysis.average_distance
                focus_position = self._calculate_basic_focus(avg_distance)

                return StateAction(
                    speak=self.voice_mapper.get_message("perfect_position"),
                    gimbal_target=(yaw_adjustment, pitch_adjustment, 0.0),
                    gimbal_time=2.0,
                    focus_position=focus_position
                )
        elif time_in_state > 3.0:
            # camera adjustments complete
            self._transition_to(PhotoState.VERIFYING, "camera_adjustment_complete")
        return None

    def _handle_verifying(self) -> Optional[StateAction]:
        """handle verifying state -- final check before capture"""
        time_in_state = self.time_in_state()

        if time_in_state < 0.5:
            return StateAction()

        #basic verification checks
        person_count = len(self.context.person_positions or [])
        if person_count == 0:
            self._transition_to(PhotoState.DETECTING, "people_disappeared_during_verification")
            return StateAction(speak=self.voice_mapper.get_message("welcome"))

        #check if people all are still positioned
        if time_in_state > 2.0:
            self._transition_to(PhotoState.COUNTDOWN, "verification_passed")
            return StateAction(speak="Perfect! Get ready...")

        return None

    def _handle_countdown(self) -> Optional[StateAction]:
        """handle countdown state -- 1-2-3 counts"""

        time_in_state = self.time_in_state()
        countdown_duration = self.config.workflow.capture_countdown_s

        #cal countdown number
        remaining_time = countdown_duration - time_in_state
        countdown_number = int(remaining_time) + 1

        if countdown_number > countdown_duration:
            # just started countdown
            return StateAction(speak=f"Photo in {countdown_duration}")

        elif countdown_number > 0 and remaining_time > 0.5:
            #count down
            return StateAction(speak=str(countdown_number))

        elif remaining_time <= 0:
            #take photo
            self._transition_to(PhotoState.CAPTURING, "countdown_complete")
            return StateAction(
                speak="Say cheese!",
                capture_photo=True
            )

        return None

    def _handle_capturing(self) -> Optional[StateAction]:
        """handle capturing state -- taking photo"""

        time_in_state = self.time_in_state()

        if time_in_state > 2.0:

            #photo capture complete
            self.context.photos_captured += 1
            self._transition_to(PhotoState.COMPLETE, "photo_captured")
            return StateAction(speak=self.voice_mapper.get_message("capture_complete"))

        return None

    def _handle_complete(self) -> Optional[StateAction]:
        """handle complete state -- session finished"""

        time_in_state = self.time_in_state()

        if time_in_state < 3.0:
            #session complete, return idle
            self._finish_session()
            self._transition_to(PhotoState.IDLE, "session_complete")

        return None

    def _handle_error(self) -> Optional[StateAction]:
        """handle error state -- ERROR occurred"""

        time_in_state = self.time_in_state()

        if time_in_state < 0.5:
            error_message = "I'm having trouble. Let me try again..."
            return StateAction(speak=error_message)

        elif time_in_state > 3.0:
            #try to recover
            self.context.consecutive_failures = 0
            self.context.last_error = None
            self._transition_to(PhotoState.DETECTING, "error_recovery_attempt")
        return None

    def _handle_timeout(self) -> Optional[StateAction]:
        """handle timeout state -- timed out occurred"""

        time_in_state = self.time_in_state()

        if time_in_state < 0.5:
            return StateAction(speak=self.voice_mapper.get_message("timeout_warning"))

        elif time_in_state > 2.0:
            # return --> detecting
            self._transition_to(PhotoState.DETECTING, "timeout_recovery")

        return None

    def _check_timeout(self) -> bool:
        """check if current state has timed out"""
        if self.current_state not in self.state_timeouts:
            return False

        timeout = self.state_timeouts[self.current_state]
        time_in_state = self.time_in_state()

        if time_in_state > timeout:
            logger.warning("state_timeout",
                state=self.current_state.name,
                time_in_state=time_in_state,
                timeout=timeout)

            self._transition_to(PhotoState.TIMEOUT, f"timeout_after_{timeout}s")
            return True

        return False

    def _transition_to(
            self,
            new_state: PhotoState,
            reason: str
    ) -> None:
        """transition state to new state"""

        old_state = self.current_state
        self.previous_state = old_state
        self.current_state = new_state
        self.state_entry_time = time.time()

        #reset state-specific context data
        if new_state == PhotoState.DETECTING:
            self.context.stable_detection_start = None
        elif new_state == PhotoState.POSITIONING:
            self.context.guidance_attempts = 0
            self.context.last_guidance_time = 0.0

        self.session_stats['state_transitions'] += 1

        logger.info("state_transition",
                    from_state=old_state.name,
                    to_state=new_state.name,
                    reason=reason,
                    time_in_previous_state=time.time() - self.state_entry_time)

    def _classify_basic_scene(self, positions: List[PersonPosition]) -> SceneType:
        """Basic scene classification."""

        #TODO: later implement advanced classifications

        count = len(positions)

        if count == 1:
            return SceneType.PORTRAIT
        elif count == 2:
            # check if close together
            pos1, pos2 = positions[0].position_3d, positions[1].position_3d
            distance = abs(pos1[0] - pos2[0])
            return SceneType.COUPLE if distance < 1.0 else SceneType.SEPARATED_PEOPLE
        elif count <= 4:
            return SceneType.SMALL_GROUP
        elif count <= 7:
            return SceneType.MEDIUM_GROUP
        else:
            return SceneType.LARGE_GROUP

    def _analyze_basic_scene(self, positions: List[PersonPosition]) -> SceneAnalysis:
        """Basic scene analysis."""

        # TODO: later implement advanced scene analyze
        if not positions:
            return SceneAnalysis(
                scene_type=SceneType.UNKNOWN,
                person_count=0,
                confidence=0.0,
                group_centroid=(0.0, 0.0, 0.0),
                horizontal_spread=0.0,
                depth_spread=0.0,
                average_distance=0.0,
                composition_quality=CompositionQuality.UNACCEPTABLE,
                framing_score=0.0,
                spacing_score=0.0,
                distance_score=0.0,
                optimal_distance=4.0,
                recommended_action_keys=[],
                camera_adjustments={},
                faces_detected=0,
                faces_visible=0,
                face_quality_average=0.0,
                analysis_timestamp=time.time(),
                processing_time_ms=0.0,
                positions_used=positions
            )

        #basic calculations
        pos_3d = [p.position_3d for p in positions]
        distances = [p.distance_from_camera for p in positions]

        centroid = (
            np.mean([p[0] for p in pos_3d]),
            np.mean([p[1] for p in pos_3d]),
            np.mean([p[2] for p in pos_3d])
        )

        x_positions = [p[0] for p in pos_3d]
        horizontal_spread = max(x_positions) - min(x_positions) if len(x_positions) > 1 else 0.0

        z_positions = [p[2] for p in pos_3d]
        depth_spread = max(z_positions) - min(z_positions) if len(z_positions) > 1 else 0.0

        average_distance = np.mean(distances)

        #simple quality assessment
        optimal_distances = {
            SceneType.PORTRAIT: 3.5,
            SceneType.COUPLE: 3.0,
            SceneType.SMALL_GROUP: 4.5,
            SceneType.MEDIUM_GROUP: 5.5,
            SceneType.LARGE_GROUP: 7.0
        }

        scene_type = self._classify_basic_scene(positions)
        optimal_distance = optimal_distances.get(scene_type, 4.0)

        distance_error = abs(average_distance - optimal_distance)
        distance_score = max(0.0, 1.0 - distance_error / 2.0)

        if distance_score > 0.8:
            quality = CompositionQuality.EXCELLENT
        elif distance_score > 0.6:
            quality = CompositionQuality.GOOD
        elif distance_score > 0.4:
            quality = CompositionQuality.FAIR
        else:
            quality = CompositionQuality.POOR

        return SceneAnalysis(
            scene_type=scene_type,
            person_count=len(positions),
            confidence=0.8,
            group_centroid=centroid,
            horizontal_spread=horizontal_spread,
            depth_spread=depth_spread,
            average_distance=average_distance,
            composition_quality=quality,
            framing_score=0.7,
            spacing_score=0.7,
            distance_score=distance_score,
            optimal_distance=optimal_distance,
            recommended_action_keys=[],
            camera_adjustments={},
            faces_detected=len(positions),
            faces_visible=len(positions),
            face_quality_average=0.8,
            analysis_timestamp=time.time(),
            processing_time_ms=5.0,
            positions_used=positions
        )

    # def _generate_guidance_messages(self, scene_analysis: SceneAnalysis) -> str:
    #     """generate guidance messages based on scene"""
    #     avg_distance = scene_analysis.average_distance
    #     optimal_distance = scene_analysis.optimal_distance
    #
    #     distance_diff = avg_distance - optimal_distance
    #
    #     if distance_diff > 0.5:
    #         return "Please step closer to me"
    #     elif distance_diff < -0.5:
    #         return "Please step back a little"
    #     elif scene_analysis.horizontal_spread > 3.0:
    #         return "Please move closer together"
    #     elif abs(scene_analysis.group_centroid[0]) > 0.5:
    #         if scene_analysis.group_centroid[0] > 0:
    #             return "Please move a bit to your left"
    #         else:
    #             return "Please move a bit to your right"
    #     else:
    #         return "Almost perfect! Small adjustment please"

    def _calculate_basic_focus(self, distance: float) -> int:
        """Basic focus calculation based on distance (returns zoom position 0-4095)."""
        # Map distance to zoom level (0=wide, 4095=telephoto)
        # For photography: closer subjects → zoom in more (higher values)
        if distance < 2.0:
            return 4095  # Maximum zoom for very close subjects (< 2m)
        elif distance < 3.0:
            return 3072  # 75% zoom for close subjects (2-3m)
        elif distance < 5.0:
            return 2048  # 50% zoom for medium distance (3-5m)
        elif distance < 8.0:
            return 1024  # 25% zoom for far subjects (5-8m)
        else:
            return 0  # Wide angle for very far subjects (>8m)

        # TODO: implement advanced distance based focus logic --> use realsense

    def _finish_session(self) -> None:
        """complete session and update stats"""
        session_duration = time.time() - self.context.session_start_time

        self.session_stats['sessions_completed'] += 1
        self.session_stats['total_session_time'] += session_duration

        if self.session_stats['sessions_completed'] > 0:
            avg_time = self.session_stats['total_session_time'] / self.session_stats['sessions_completed']
            self.session_stats['average_session_time'] = avg_time

        logger.info("session_completed",
                    duration_s=session_duration,
                    photos_captured=self.context.photos_captured,
                    guidance_attempts=self.context.guidance_attempts)

        # reset context for next session
        self.context = StateContext()

    def _create_action_dict(self, action: Optional[StateAction] = None) -> Dict[str, Any]:
        """create action dict"""

        if action is None:
            return {}

        result = {}

        result = {}

        if action.speak:
            result['speak'] = action.speak

        if action.gimbal_target:
            result['gimbal_target'] = action.gimbal_target
            result['gimbal_time'] = action.gimbal_time

        if action.focus_position is not None:
            result['focus_position'] = action.focus_position

        if action.capture_photo:
            result['capture_photo'] = True

        return result

    def time_in_state(self) -> float:
        """get time elapsed in current state"""
        return time.time() - self.state_entry_time

    def get_session_stats(self) -> Dict[str, Any]:
        """get current session statistics."""
        return {
            **self.session_stats,
            'current_state': self.current_state.name,
            'time_in_current_state': self.time_in_state(),
            'photos_captured_this_session': self.context.photos_captured,
            'guidance_attempts_this_session': self.context.guidance_attempts
        }





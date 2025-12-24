import time
from typing import Optional, Dict, List, Tuple
from collections import deque

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ..utils.logger import get_logger
from config.settings import get_settings
from .person_tracker import PersonTracker, PersonTrack, TrackState

logger = get_logger(__name__)

class AttentionManager:

    def __init__(self, person_tracker: PersonTracker) -> None:
        self.person_tracker = person_tracker
        settings = get_settings()
        self.config = settings.tracking.attention_management

        # current focus state
        self.current_focus_target: Optional[int] = None
        self.focus_change_cooldown_until: float = 0.0
        self.last_focus_change_time: float = 0.0

        # configs
        self.min_focus_duration = self.config.min_focus_duration
        self.min_attention_improvement = self.config.min_attention_improvement
        self.smoothing = self.config.focus_change_smoothing

        # history
        self.attention_history: deque = deque(
            maxlen=self.config.attention_history_size
        )
        self.focus_history: List[Tuple[float, int]] = []  # (timestamp, track_id)

        logger.info(
            "attention_manager_initialized",
            min_focus_duration=self.min_focus_duration,
            min_improvement=self.min_attention_improvement
        )

    def determine_optimal_focus(self) -> Optional[int]:

        current_time = time.time()

        # check if in cooldown
        if current_time < self.focus_change_cooldown_until:
            # if in cooldown --> return current target
            if getattr(self.config, 'log_focus_changes', False):
                logger.debug(
                    "focus_cooldown_active",
                    current_target=self.current_focus_target,
                    remaining=round(self.focus_change_cooldown_until - current_time, 1)
                )
            return self.current_focus_target

        # get all active tracks
        tracks = self.person_tracker.active_tracks

        # filter by --> confirmed tracks
        trackable = {
            tid: track for tid, track in tracks.items()
            if track.state == TrackState.CONFIRMED
        }

        if not trackable:
            # no suitable tracks
            if self.current_focus_target is not None:
                logger.info("no_trackable_subjects_clearing_focus")
                self.current_focus_target = None
            return None

        # find highest attention track
        best_track = max(trackable.values(), key=lambda t: t.attention_score)

        # decide if focus should change
        if self._should_change_focus(best_track, current_time):
            # Change focus
            old_target = self.current_focus_target
            self.current_focus_target = best_track.track_id
            self.last_focus_change_time = current_time
            self.focus_change_cooldown_until = current_time + self.min_focus_duration

            # log focus change
            if getattr(self.config, 'log_focus_changes', False):
                logger.info(
                    "focus_target_changed",
                    old_target=old_target,
                    new_target=best_track.track_id,
                    attention_score=round(best_track.attention_score, 3),
                    cooldown_duration=self.min_focus_duration
                )

            # record in history
            self.focus_history.append((current_time, best_track.track_id))

            # limit history size
            if len(self.focus_history) > 100:
                self.focus_history = self.focus_history[-100:]

        return self.current_focus_target

    def _should_change_focus(self, candidate_track: TrackState, current_time: float) -> bool:
        """determine if focus should be changed."""

        # if no current target --> accept any candidate
        if self.current_focus_target is None:
            return True

        # get current target track
        current_track = self.person_tracker.get_track(self.current_focus_target)

        # if current track is lost/ gone --> accept new candidate
        if current_track is None:
            return True

        # if current track is no longer confirmed --> accept new candidate
        if current_track.state != TrackState.CONFIRMED:
            return True

        # check attention score difference
        attention_diff = candidate_track.attention_score - current_track.attention_score

        # require significant improvement to change focus --> hysteresis
        if attention_diff > self.min_attention_improvement:
            return True

        # check if current track is significantly worse than candidate --> emergency focus change if current track dropped a lot
        if current_track.attention_score < 0.3 and candidate_track.attention_score > 0.6:
            logger.info(
                "emergency_focus_change",
                reason="current_track_attention_too_low",
                current_score=current_track.attention_score,
                candidate_score=candidate_track.attention_score
            )
            return True

            # do not change focus
        return False

    def get_attention_distribution(self) -> Dict[int, float]:
        """get normalized attention distribution across all confirmed tracks"""

        tracks = {
            tid: track for tid, track in self.person_tracker.active_tracks.items()
            if track.state == TrackState.CONFIRMED
        }

        if not tracks:
            return {}

        # calculate total attention
        total_attention = sum(track.attention_score for track in tracks.values())

        if total_attention == 0:
            # equal distribution if no attention scores
            equal_weight = 1.0 / len(tracks)
            return {track_id: equal_weight for track_id in tracks.keys()}

        # normalize --> sum to 1.0
        return {
            track_id: track.attention_score / total_attention
            for track_id, track in tracks.items()
        }

    def get_focus_target_info(self) -> Optional[Dict]:
        """get info about current focus target"""

        if self.current_focus_target is None:
            return None

        track = self.person_tracker.get_track(self.current_focus_target)
        if track is None:
            return None

        current_time = time.time()
        focus_duration = current_time - self.last_focus_change_time
        cooldown_remaining = max(0.0, self.focus_change_cooldown_until - current_time)

        return {
            'track_id': track.track_id,
            'attention_score': track.attention_score,
            'tracking_quality': track.tracking_quality,
            'stability_score': track.stability_score,
            'state': track.state.value,
            'focus_duration': focus_duration,
            'cooldown_remaining': cooldown_remaining,
            'can_change': cooldown_remaining == 0.0
        }

    def force_focus_change(self, track_id: int) -> bool:
        """force to change focus to specified track"""

        track = self.person_tracker.get_track(track_id)
        if track is None or track.state != TrackState.CONFIRMED:
            logger.warning(f"cannot_force_focus: track {track_id} not found or not confirmed")
            return False

        old_target = self.current_focus_target
        current_time = time.time()

        self.current_focus_target = track_id
        self.last_focus_change_time = current_time
        self.focus_change_cooldown_until = current_time + self.min_focus_duration

        logger.info(
            "focus_forced",
            old_target=old_target,
            new_target=track_id
        )

        self.focus_history.append((current_time, track_id))

        return True

    def clear_focus(self) -> None:
        """clear current focus target"""
        if self.current_focus_target is not None:
            logger.info(f"focus_cleared: was {self.current_focus_target}")
            self.current_focus_target = None
            self.focus_change_cooldown_until = 0.0

    def get_focus_statistics(self) -> Dict:
        """get focus stats"""
        if not self.focus_history:
            return {
                'total_focus_changes': 0,
                'current_focus': self.current_focus_target
            }

        # calculate focus durations
        focus_durations = []
        for i in range(1, len(self.focus_history)):
            duration = self.focus_history[i][0] - self.focus_history[i - 1][0]
            focus_durations.append(duration)

        # count focus changes per track
        track_focus_counts = {}
        for _, track_id in self.focus_history:
            track_focus_counts[track_id] = track_focus_counts.get(track_id, 0) + 1

        return {
            'total_focus_changes': len(self.focus_history),
            'current_focus': self.current_focus_target,
            'avg_focus_duration': sum(focus_durations) / len(focus_durations) if focus_durations else 0.0,
            'min_focus_duration': min(focus_durations) if focus_durations else 0.0,
            'max_focus_duration': max(focus_durations) if focus_durations else 0.0,
            'track_focus_counts': track_focus_counts
        }

    def get_performance_stats(self) -> Dict:
        """get attention manager performance stats"""
        return {
            'current_focus_target': self.current_focus_target,
            'focus_changes': len(self.focus_history),
            'min_focus_duration': self.min_focus_duration,
            'min_attention_improvement': self.min_attention_improvement
        }

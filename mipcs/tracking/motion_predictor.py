import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ..utils.logger import get_logger
from config.settings import get_settings
from .person_tracker import PersonTracker, PersonTrack, TrackState

logger = get_logger(__name__)

class MotionPredictor:

    def __init__(self, person_tracker: PersonTracker) -> None:
        self.person_tracker = person_tracker
        settings = get_settings()
        self.config = settings.tracking.motion_prediction

        # Kalman Filters --> one per track
        self.kalman_filters: Dict[int, cv2.KalmanFilter] = {}

        # configs
        self.prediction_horizon = self.config.prediction_horizon
        self.min_history = self.config.min_history_for_prediction
        self.process_noise = self.config.kalman_process_noise
        self.measurement_noise = self.config.kalman_measurement_noise
        self.confidence_decay = self.config.confidence_decay
        self.update_rate = self.config.update_rate

        logger.info(
            "motion_predictor_initialized",
            horizon=self.prediction_horizon,
            min_history=self.min_history
        )

    def predict_group_movement(
            self,
            prediction_horizon: Optional[float] = None,
    ) -> Dict:
        """Predict movement of the entire track group"""

        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon

        #get trackable subjects
        active_tracks = self.person_tracker.get_group_tracking_targets()

        if not active_tracks:
            return {
                'group_center_prediction': (0.0, 0.0, 5.0),
                'group_spread_prediction': 0.0,
                'individual_predictions': {},
                'confidence': 0.0,
                'prediction_horizon': prediction_horizon
            }

        # predict individual positions
        individual_predictions = {}
        confidence_scores = []
        valid_predictions = []

        for track in active_tracks:
            prediction, confidence = self._predict_individual_movement(
                track,
                prediction_horizon
            )

            if prediction is not None:
                individual_predictions[track.track_id] = prediction
                confidence_scores.append(confidence)
                valid_predictions.append(prediction)

        # calculate group center prediction
        if valid_predictions:
            predictions_array = np.array(valid_predictions)
            group_center = tuple(np.mean(predictions_array, axis=0))

            # calculate group spread --> max distance from center
            distances = [
                np.linalg.norm(pred - np.array(group_center))
                for pred in valid_predictions
            ]
            group_spread = max(distances) if distances else 0.0

            # overall confidence --> average of individual confidences
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        else:
            # no valid predictions
            group_center = (0.0, 0.0, 5.0)
            group_spread = 0.0
            overall_confidence = 0.0

        return {
            'group_center_prediction': group_center,
            'group_spread_prediction': group_spread,
            'individual_predictions': individual_predictions,
            'confidence': overall_confidence,
            'prediction_horizon': prediction_horizon,
            'num_predicted': len(valid_predictions)
        }

    def _predict_individual_movement(
            self,
            track: PersonTrack,
            prediction_horizon: float,
    ) -> Tuple[Optional[Tuple[float, float, float]], float]:
        """Predict movement for individual using KF"""

        # check if we have enough history
        if len(track.position_history) < self.min_history:
            # insufficient data --> return last known position with low confidence
            if track.position_history:
                return track.position_history[-1], 0.2
            else:
                return None, 0.0

        track_id = track.track_id

        # init KF for this track if needed
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = self._initialize_kalman_filter()

        kf = self.kalman_filters[track_id]

        try:
            # update KF with recent positions
            recent_positions = list(track.position_history)[-self.min_history:]

            for pos in recent_positions:
                measurement = np.array([[pos[0]], [pos[1]], [pos[2]]], dtype=np.float32)
                kf.correct(measurement)

            # predict future state  --> KF predicts only one step ahead, multiple predictions are needed for longer horizons.
            num_steps = int(prediction_horizon * self.update_rate)  # Assumption --> update rate = 30 Hz TODO: check the real update rate change if needed

            for _ in range(num_steps):
                kf.predict()

            # get predicted state
            predicted_state = kf.statePost
            predicted_position = (
                float(predicted_state[0, 0]),
                float(predicted_state[1, 0]),
                float(predicted_state[2, 0])
            )

            # calculate confidence based on:        # TODO: Research more KF
            #           1. Tracking quality
            #           2. Movement stability
            #           3. Decay with prediction horizon
            base_confidence = track.tracking_quality * track.stability_score
            horizon_decay = self.confidence_decay ** prediction_horizon
            confidence = base_confidence * horizon_decay

            return predicted_position, confidence

        except Exception as e:
            logger.warning(f"prediction_failed for track {track_id}: {e}")
            # fall back --> last known position
            if track.position_history:
                return track.position_history[-1], 0.1
            return None, 0.0

    def _initialize_kalman_filter(self) -> cv2.KalmanFilter:
        """Initialize kalman filter"""

        # create filter --> 6 state variables, 3 measurements
        kf = cv2.KalmanFilter(6, 3)

        # transition matrix -> constant velocity model
        dt = 1.0/ self.update_rate   # Assumption --> update rate = 30 Hz TODO: check the real update rate change if needed

        kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # measurement matrix --> observe position only
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # process noise covariance --> model trust level
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * self.process_noise

        # measurement noise covariance --> measurement trust level
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * self.measurement_noise

        # error covariance matrix --> initial uncertainty
        kf.errorCovPost = np.eye(6, dtype=np.float32)

        # initial state
        kf.statePost = np.zeros((6, 1), dtype=np.float32)

        return kf

    def predict_track_position(
            self,
            track_id: int,
            prediction_horizon: Optional[float] = None
    ) -> Optional[Tuple[float, float, float]]:
        """predict position of the track"""

        track = self.person_tracker.get_track(track_id)
        if track is None:
            return None

        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon

        prediction, _ = self._predict_individual_movement(track, prediction_horizon)
        return prediction

    def cleanup_filters(self) -> None:
        """cleanup filters"""
        active_track_ids = set(self.person_tracker.active_tracks.keys())
        filter_track_ids = set(self.kalman_filters.keys())

        # remove filters for dead tracks
        dead_tracks = filter_track_ids - active_track_ids
        for track_id in dead_tracks:
            del self.kalman_filters[track_id]

        if dead_tracks:
            logger.debug(f"cleaned_up_filters: removed {len(dead_tracks)} filters")

    def get_performance_stats(self) -> Dict:
        """get motion prediction performance stats"""
        return {
            'active_filters': len(self.kalman_filters),
            'prediction_horizon': self.prediction_horizon,
            'min_history_required': self.min_history
        }


import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ..utils.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


class TrackState(Enum):
    """Track lifecycle states"""
    TENTATIVE = "tentative"  # New track, needs confirmation
    CONFIRMED = "confirmed"  # Stable, reliable track
    OCCLUDED = "occluded"  # Temporarily lost (predicted position)
    LOST = "lost"  # Track expired, will be removed


@dataclass
class PersonTrack:
    """Individual person track with full state"""
    track_id: int
    state: TrackState

    # position tracking
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # quality metrics
    attention_score: float = 0.5
    tracking_quality: float = 0.5
    stability_score: float = 0.5

    # temporal data
    created_at: float = 0.0
    last_seen: float = 0.0
    last_updated: float = 0.0

    # detection association
    last_detection_confidence: float = 0.0
    consecutive_hits: int = 0
    consecutive_misses: int = 0

    # face and pose quality
    face_quality: float = 0.0
    pose_quality: float = 0.0
    facing_camera: bool = False


class PersonTracker:

    def __init__(self):
        settings = get_settings()
        self.config = settings.tracking

        # track management
        self.active_tracks: Dict[int, PersonTrack] = {}
        self.next_track_id = 1

        # config params
        self.max_tracking_distance = self.config.max_tracking_distance
        self.max_association_cost = self.config.max_association_cost
        self.track_timeout = self.config.track_timeout
        self.occlusion_timeout = self.config.occlusion_timeout
        self.lost_track_timeout = self.config.lost_track_timeout
        self.min_confidence = self.config.min_confidence
        self.min_tracking_quality = self.config.min_tracking_quality
        self.tentative_confirmation = self.config.tentative_confirmation

        # attention scoring weights
        self.attention_weights = self.config.attention_weights
        self.optimal_distance_min = self.config.optimal_distance_min
        self.optimal_distance_max = self.config.optimal_distance_max

        # performance tracking
        self.tracks_created = 0
        self.tracks_lost = 0
        self.total_tracking_time = 0.0

        logger.info(
            "person_tracker_initialized",
            max_distance=self.max_tracking_distance,
            timeout=self.track_timeout
        )

    async def update_tracks(
            self,
            person_detections: List[Dict],
            positions_3d: Dict[int, Tuple[float, float, float]],
            face_analyses: Optional[List[Dict]] = None,
            timestamp: Optional[float] = None,
    ) -> Dict[int, PersonTrack]:
        """Update person tracks"""

        if timestamp is None:
            timestamp = time.time()

        start_time = time.time()

        try:
            # 1. associate detections with existing tracks
            associations = self._associate_detections_to_tracks(
                person_detections,
                positions_3d,
                timestamp
            )

            # 2. update existing tracks with associated detections
            self._update_existing_tracks(
                associations,
                person_detections,
                positions_3d,
                face_analyses,
                timestamp
            )

            # 3. create new tracks for unassociated detections
            self._create_new_tracks(
                person_detections,
                positions_3d,
                face_analyses,
                associations,
                timestamp
            )

            # 4. manage track lifecycle --> handle occlusions, timeouts
            self._manage_track_lifecycle(timestamp)

            # 5. update attention scores
            self._update_attention_scores(timestamp)

            # track performance
            elapsed = time.time() - start_time
            self.total_tracking_time += elapsed

            logger.debug(
                "tracks_updated",
                active_tracks=len(self.active_tracks),
                new_detections=len(person_detections),
                time_ms=round(elapsed * 1000, 1)
            )

            return self.active_tracks.copy()

        except Exception as e:
            logger.error(f"track_update_failed: {e}", exc_info=True)
            return self.active_tracks.copy()

    def _associate_detections_to_tracks(
            self,
            detections: List[Dict],
            positions_3d: Dict[int, Tuple[float, float, float]],
            timestamp: float
    ) -> Dict[int, int]:
        """Associate new detections with existing tracks --> use Hungarian algorithm"""

        associations = {}

        if not self.active_tracks or not detections:
            return associations

        # filter trackable tracks (not lost)
        trackable = {
            tid: track for tid, track in self.active_tracks.items()
            if track.state != TrackState.LOST
        }

        if not trackable:
            return associations

        # build cost matrix for hungarian algorithm
        track_ids = list(trackable.keys())
        detection_indices = list(range(len(detections)))

        num_tracks = len(track_ids)
        num_detections = len(detection_indices)

        # init cost matrix with high values
        cost_matrix = np.full(
            (num_tracks, num_detections),
            fill_value=self.max_association_cost * 2
        )

        # calculate costs -> distance based
        for i, track_id in enumerate(track_ids):
            track = trackable[track_id]

            if not track.position_history:
                continue

            # get last known position
            last_pos = track.position_history[-1]

            for j, det_idx in enumerate(detection_indices):
                if det_idx not in positions_3d:
                    continue

                detection_pos = positions_3d[det_idx]

                # cal euclidean distance
                distance = np.linalg.norm(
                    np.array(detection_pos) - np.array(last_pos)
                )

                # consider only if within max distance
                if distance < self.max_tracking_distance:
                    # cost is distance
                    quality_factor = 1.0 - (track.tracking_quality * 0.3)
                    cost_matrix[i, j] = distance * quality_factor

        # run hungarian algorithm
        if num_tracks > 0 and num_detections > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            # build associations --> filter by max cost
            for track_idx, det_idx in zip(row_indices, col_indices):
                if cost_matrix[track_idx, det_idx] < self.max_association_cost:
                    track_id = track_ids[track_idx]
                    detection_index = detection_indices[det_idx]
                    associations[track_id] = detection_index

        return associations

    def _update_existing_tracks(
            self,
            associations: Dict[int, int],
            detections: List[Dict],
            positions_3d: Dict[int, Tuple[float, float, float]],
            face_analyses: Optional[List[Dict]],
            timestamp: float
    ) -> None:
        """Update tracks that were successfully associated with detections"""

        for track_id, det_idx in associations.items():
            track = self.active_tracks[track_id]

            if det_idx not in positions_3d:
                continue

            # Update position
            new_position = positions_3d[det_idx]
            track.position_history.append(new_position)

            # Update velocity (if we have history)
            if len(track.position_history) >= 2:
                prev_pos = track.position_history[-2]
                dt = timestamp - track.last_updated
                if dt > 0:
                    velocity = tuple(
                        (new_position[i] - prev_pos[i]) / dt
                        for i in range(3)
                    )
                    # Smooth velocity
                    smoothing = 0.5
                    track.velocity = tuple(
                        track.velocity[i] * (1 - smoothing) + velocity[i] * smoothing for i in range(3)
                    )

            # update quality metrics
            detection = detections[det_idx]
            track.last_detection_confidence = detection.get('confidence', 0.5)

            # update face quality --> if available
            if face_analyses and det_idx < len(face_analyses):
                face = face_analyses[det_idx]
                if face:
                    track.face_quality = face.get('confidence', 0.0)
                    track.facing_camera = face.get('facing_camera', False)

            # update pose quality
            track.pose_quality = detection.get('pose_quality', 0.5)

            # update temporal data
            track.last_seen = timestamp
            track.last_updated = timestamp
            track.consecutive_hits += 1
            track.consecutive_misses = 0

            # update tracking quality
            track.tracking_quality = self._calculate_tracking_quality(track)

            # update stability
            track.stability_score = self._calculate_stability(track)

            # confirm tentative tracks
            if track.state == TrackState.TENTATIVE:
                if track.consecutive_hits >= self.tentative_confirmation:
                    track.state = TrackState.CONFIRMED
                    logger.info(f"track_confirmed: track_id={track_id}")

            # restore occluded tracks
            elif track.state == TrackState.OCCLUDED:
                track.state = TrackState.CONFIRMED
                logger.info(f"track_recovered: track_id={track_id}")


    def _create_new_tracks(
            self,
            detections: List[Dict],
            positions_3d: Dict[int, Tuple[float, float, float]],
            face_analyses: Optional[List[Dict]],
            associations: Dict[int, int],
            timestamp: float
    ) -> None:
        """create new tracks for unassociated detections"""

        # find unassociated detections
        associated_detections = set(associations.values())
        unassociated = [
            i for i in range(len(detections))
            if i not in associated_detections and i in positions_3d
        ]

        for det_idx in unassociated:
            detection = detections[det_idx]
            position = positions_3d[det_idx]

            # check minimum confidence
            if detection.get('confidence', 0) < self.min_confidence:
                continue

            # create new track
            track_id = self.next_track_id
            self.next_track_id += 1

            track = PersonTrack(
                track_id=track_id,
                state=TrackState.TENTATIVE,
                created_at=timestamp,
                last_seen=timestamp,
                last_updated=timestamp
            )

            # init position
            track.position_history.append(position)

            # init quality metrics
            track.last_detection_confidence = detection.get('confidence', 0.5)
            track.pose_quality = detection.get('pose_quality', 0.5)

            # Init face quality if available
            if face_analyses and det_idx < len(face_analyses):
                face = face_analyses[det_idx]
                if face:
                    track.face_quality = face.get('confidence', 0.0)
                    track.facing_camera = face.get('facing_camera', False)

            track.consecutive_hits = 1
            track.tracking_quality = 0.5
            track.stability_score = 0.5
            track.attention_score = 0.3  # Start low

            self.active_tracks[track_id] = track
            self.tracks_created += 1

            if self.config.log_track_creation:
                logger.info(
                    "track_created",
                    track_id=track_id,
                    position=position
                )

    def _manage_track_lifecycle(self, timestamp: float) -> None:
        """handle track timeouts and state transitions"""

        tracks_to_remove = []

        for track_id, track in self.active_tracks.items():
            time_since_seen = timestamp - track.last_seen

            # handle tentative tracks
            if track.state == TrackState.TENTATIVE:
                if time_since_seen > self.track_timeout:
                    tracks_to_remove.append(track_id)
                    continue

            # handle confirmed tracks
            elif track.state == TrackState.CONFIRMED:
                if time_since_seen > self.occlusion_timeout:
                    # Mark as occluded
                    track.state = TrackState.OCCLUDED
                    logger.info(f"track_occluded: track_id={track_id}")

            # handle occluded tracks
            elif track.state == TrackState.OCCLUDED:
                if time_since_seen > self.track_timeout:
                    # Mark as lost
                    track.state = TrackState.LOST
                    logger.info(f"track_lost: track_id={track_id}")

            # handle lost tracks
            elif track.state == TrackState.LOST:
                if time_since_seen > self.lost_track_timeout:
                    tracks_to_remove.append(track_id)

        # remove expired tracks
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
            self.tracks_lost += 1

            if self.config.log_track_loss:
                logger.info(f"track_removed: track_id={track_id}")

    def _update_attention_scores(self, timestamp: float) -> None:
        """update attention scores for all tracks"""

        for track in self.active_tracks.values():
            if track.state == TrackState.LOST:
                track.attention_score = 0.0
                continue

            # component scores
            face_score = track.face_quality * self.attention_weights.face_quality
            pose_score = track.pose_quality * self.attention_weights.pose_quality

            # distance score [optimal range 2 - 6]  # TODO: Check again after ZED X is used.
            if track.position_history:
                distance = track.position_history[-1][2]  # z coord
                if self.optimal_distance_min <= distance <= self.optimal_distance_max:
                    distance_score = 1.0
                else:
                    # penalty --> for being outside optimal range
                    if distance < self.optimal_distance_min:
                        distance_score = distance / self.optimal_distance_min
                    else:
                        distance_score = self.optimal_distance_max / distance
                distance_score *= self.attention_weights.distance_optimal
            else:
                distance_score = 0.0

            # stability score
            stability_score = track.stability_score * self.attention_weights.stability

            # duration score (longer tracking = higher attention)
            duration = timestamp - track.created_at
            duration_score = min(1.0, duration / 10.0) * self.attention_weights.tracking_duration

            # combine scores
            track.attention_score = (
                    face_score +
                    pose_score +
                    distance_score +
                    stability_score +
                    duration_score
            )

            # bonus for facing camera
            if track.facing_camera:
                track.attention_score *= 1.2

            # clamp to [0, 1]
            track.attention_score = np.clip(track.attention_score, 0.0, 1.0)

    def calculate_tracking_quality(self, track: PersonTrack) -> float:
        """calculate tracking quality"""

        # recent hit rate
        hit_rate = track.consecutive_hits / max(1, track.consecutive_hits + track.consecutive_misses)

        # detection conf
        confidence = track.last_detection_confidence

        # position consistency  (low velocity = higher quality)
        velocity_magnitude = np.linalg.norm(track.velocity)
        velocity_factor = 1.0 / (1.0 + velocity_magnitude)

        # combine
        quality = (hit_rate * 0.5 + confidence * 0.3 + velocity_factor * 0.2)

        return np.clip(quality, 0.0, 1.0)

    def _calculate_stability(self, track: PersonTrack) -> float:
        """calculate movement stability score"""
        if len(track.position_history) < 3:
            return 0.5

            # calculate position variance
        positions = np.array(list(track.position_history))
        center = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in positions]
        spread = np.std(distances)

        # lower spread = higher stability
        stability = 1.0 / (1.0 + spread)

        return np.clip(stability, 0.0, 1.0)

    def get_group_tracking_targets(self) -> List[PersonTrack]:
        """get tracks suitable for group framing"""

        confirmed = [
            track for track in self.active_tracks.values()
            if track.state == TrackState.CONFIRMED
        ]

        # sort by attention score
        confirmed.sort(key=lambda t: t.attention_score, reverse=True)

        return confirmed

    def get_track(self, track_id: int) -> Optional[PersonTrack]:
        """get track by track id"""
        return self.active_tracks.get(track_id)

    def get_performance_stats(self) -> Dict:
        """get tracking performance stats"""
        return {
            'active_tracks': len(self.active_tracks),
            'tracks_created': self.tracks_created,
            'tracks_lost': self.tracks_lost,
            'avg_tracking_time_ms': (
                    (self.total_tracking_time / max(1, self.tracks_created)) * 1000
            ),
            'confirmed_tracks': len([
                t for t in self.active_tracks.values()
                if t.state == TrackState.CONFIRMED
            ]),
            'tentative_tracks': len([
                t for t in self.active_tracks.values()
                if t.state == TrackState.TENTATIVE
            ])
        }








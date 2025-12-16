import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import threading
from datetime import datetime, timedelta
import sys
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
    from geometry_msgs.msg import TransformStamped
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False
    print("Warning: tf2_ros not available. Using fallback mode.")

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ..utils.logger import get_logger
logger = get_logger(__name__)

@dataclass
class Point3D:
    """3D point representation"""
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_homogeneous(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, 1.0])

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'Point3D':
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    @classmethod
    def from_homogeneous(cls, arr: np.ndarray) -> 'Point3D':
        if arr[3] != 0:
            return cls(float(arr[0] / arr[3]), float(arr[1] / arr[3]), float(arr[2] / arr[3]))
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

class TransformManager:

    def __init__(
            self, node: Optional[Node] = None,
            cache_duration: float = 1.0,
            use_tf: bool = True
    ):
        self.use_tf = use_tf
        self.cache_duration = cache_duration

        # transform cache
        self._cache: Dict[Tuple[str, str], Tuple[np.ndarray, datetime]] = {}
        self._cache_lock = threading.Lock()

        if self.use_tf:
            if node is None:
                raise ValueError("Node required when use_tf=True")
            self.node = node
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, node)
            logger.info("Transform Manager initialized with TF2")
        else:
            self.node = None
            self.tf_buffer = None
            self.tf_listener = None
            # fallback -> default Manriix values
            self._default_camera_to_base = self._create_default_transform()
            if node:
                logger.warning("Transform Manager using fallback mode (no TF2)")


    def _create_default_transform(self) -> np.ndarray:
        """create default transform matricx --> FALLBACK OPTION"""

        R = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        t = np.array([0.2005, 0.0, 0.72354])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_transform_matrix(
            self,
            source_frame: str,
            target_frame: str,
            timeout: float = 1.0
    ) -> Optional[np.ndarray]:
        """get transform matrix from source_frame to target_frame"""

        #1. check cache
        cache_key = (source_frame, target_frame)
        with self._cache_lock:
            if cache_key in self._cache:
                matrix, timestamp = self._cache[cache_key]
                age = (datetime.now() - timestamp).total_seconds()
                if age < self.cache_duration:
                    return matrix

        #fetch new transform
        if self.use_tf:
            matrix = self._lookup_transform_from_tf(source_frame, target_frame, timeout)
        else:
            # MODE = fallback
            if source_frame == "camera_color_optical_frame" and target_frame == "base_link":
                matrix = self._default_camera_to_base
            else:
                matrix = None

        # update cache
        if matrix is not None:
            with self._cache_lock:
                self._cache[cache_key] = (matrix, datetime.now())

        return matrix

    def _lookup_transform_from_tf(
            self,
            source_frame: str,
            target_frame: str,
            timeout: float
    ) -> Optional[np.ndarray]:
        """look up transformation from tf2"""

        try:
            transform_stamped: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),  #Latest
                timeout = rclpy.duration.Duration(seconds=timeout)
            )

            #convert to homogeneous matrix
            matrix = self._transform_stamped_to_matrix(transform_stamped)

            logger.debug(f"TF lookup: {source_frame} → {target_frame} succeeded")

            return matrix

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            logger.warning( f"TF lookup failed: {source_frame} → {target_frame}: {e}")
            return None


    def _transform_stamped_to_matrix(
            self,
            transform_stamped: TransformStamped
    )-> np.ndarray:
        """Convert ROS TransformStamped to 4x4 homogeneous matrix"""

        # extract translation
        t = transform_stamped.transform.translation
        translation = np.array([t.x, t.y, t.z])

        # extract rotation
        q = transform_stamped.transform.rotation
        rotation = self._quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)

        # build homogeneous matrix
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation

        return T

    def _quaternion_to_rotation_matrix(
            self,
            x: float,
            y: float,
            z: float,
            w: float
    ) -> np.ndarray:
        """convert quaternion to 3 * 3 rotation matrix"""

        # normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            x, y, z, w = x / norm, y / norm, z / norm, w / norm

        # convert to rotation matrix
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        ])

        return R

    def transform_point(
            self,
            point: Point3D,
            source_frame: str,
            target_frame: str,
    ) -> Optional[Point3D]:
        """transform point from source_frame to target_frame"""

        T = self.get_transform_matrix(source_frame, target_frame)
        if T is None:
            return None

        # apply transformation
        p_h = point.to_homogeneous()
        p_transformed_h = T @ p_h

        return Point3D.from_homogeneous(p_transformed_h)

    def transform_points_batch(
            self,
            points: list[Point3D],
            source_frame: str,
            target_frame: str,
    ) -> Optional[list[Point3D]]:
        """Transform multiple points"""

        T = self.get_transform_matrix(source_frame, target_frame)
        if T is None:
            return None

        if not points:
            return []

        # vectorized transformation
        points_h = np.array([p.to_homogeneous() for p in points])
        points_transformed_h = (T @ points_h.T).T

        return [Point3D.from_homogeneous(p) for p in points_transformed_h]


    def get_camera_position(
            self,
            camera_frame: str = "camera_color_optical_frame",
            base_frame: str = "base_link"
    ) -> Optional[Point3D]:
        """get camera position in base frame"""

        T = self.get_transform_matrix(camera_frame, base_frame)
        if T is None:
            return None

        # camera position = translation
        return Point3D(T[0, 3], T[1, 3], T[2, 3])

    def invalidate_cache(
            self,
            source_frame: Optional[str] = None,
            target_frame: Optional[str] = None
    ):
        """Invalidate cache transforms"""
        with self._cache_lock:
            if source_frame is None and target_frame is None:
                # Clear all cache
                self._cache.clear()
            else:
                # Selective invalidation
                keys_to_remove = [
                    key for key in self._cache.keys()
                    if (source_frame is None or key[0] == source_frame) and (target_frame is None or key[1] == target_frame)
                ]
                for key in keys_to_remove:
                    del self._cache[key]

    def get_cache_stats(self) -> Dict[str, int]:
        """get cache statistics"""
        with self._cache_lock:
            return {
                'cached_transforms': len(self._cache),
                'cache_duration_sec': self.cache_duration
            }




def camera_optical_to_base_link(
        camera_point: Point3D,
        transform_manager: TransformManager,
        camera_frame: str = "camera_color_optical_frame",
        base_frame: str = "base_link"
) -> Optional[Point3D]:
    """transform point from camera optical frame to base link"""

    return transform_manager.transform_point(camera_point, camera_frame, base_frame)

def base_link_to_camera_optical(
        base_point: Point3D,
        transform_manager: TransformManager,
        camera_frame: str = "camera_color_optical_frame",
        base_frame: str = "base_link"
) -> Optional[Point3D]:
    """transform point from base link to camera optical frame"""

    return transform_manager.transform_point(base_point, base_frame, camera_frame)







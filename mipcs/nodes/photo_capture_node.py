import asyncio
import time
import threading
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

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ..vision.detector import YOLOPoseDetector, PersonDetection
from ..vision.face_analyzer import MediaPipeFaceAnalyzer, FaceAnalysis
from ..vision.frame_client import CanonFrameClient, FrameMetadata
from ..vision.depth_processor import DepthProcessor, DepthData

from ..positioning.po
import asyncio
import socket
import struct
from logging.config import stopListening
from pathlib import Path
import sys
import numpy as np
import cv2
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings, Settings
from ..utils.logger import get_logger, log_performance
from ..utils.threading_utils import AsyncThreadSafeQueue, get_thread_manager, async_retry
from ..utils.exceptions import (
    FrameClientError,
    CameraConnectionError,
    NetworkError,
    TimeoutError as ManriixTimeoutError,
)

logger = get_logger(__name__)

class ConnectionState(Enum):
    """Frame client connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECEIVING = "receiving"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class FrameMetadata:
    """Metadata for received frames."""
    timestamp: float
    frame_id: int
    size: tuple
    fps: float
    latency_ms: float

class FPSCalculator:
    """Calculate fps average"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []

    def update(self, timestamp: float) -> float:
        """Update with new timestamp and return current FPS."""
        self.timestamps.append(timestamp)

        # Keep only recent timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

        return self.get_fps()

    def get_fps(self) -> float:
        """Get current FPS calculation."""
        if len(self.timestamps) < 2:
            return 0.0

        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span <= 0:
            return 0.0

        return (len(self.timestamps) - 1) / time_span

class CanonFrameClient:

    def __init__(self, frame_callback: Optional[Callable[[np.ndarray, FrameMetadata], None]] = None):
        self.settings = get_settings()
        self.config = self.settings.vision.canon

        #Connection state
        self.state = ConnectionState.DISCONNECTED
        self.socket: Optional[socket.socket] = None
        self.frame_callback = frame_callback

        #frame buffering
        self.frame_queue = AsyncThreadSafeQueue(maxsize=self.settings.vision.canon.frame_queue_size)
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_metadata: Optional[FrameMetadata] = None

        #Threading
        self.receiver_task: Optional[asyncio.Task] = None
        self.running = False
        self.shutdown_event = asyncio.Event()

        #performance tracking
        self.frames_received = 0
        self.bytes_received = 0
        self.last_frame_time = 0.0
        self.fps_calculator = FPSCalculator()

        #Reconnection handling
        self.reconnect_attempts = 0
        self.last_reconnect_time = 0.0

        logger.info(
            "canon_frame_client_initialized",
            host=self.config.host,
            port=self.config.port,
            queue_size=self.config.frame_queue_size
        )

    async def start(self) -> None:
        """Start the frame client"""
        if self.running:
            logger.warning("frame_client_already_running")
            return

        self.running = True
        self.shutdown_event.clear()

        #start receiver task
        self.receiver_task = asyncio.create_task(self._receiver_loop())

        logger.info("frame_client_started")

    async def stop(self) -> None:
        """Stop frame client"""
        if not self.running:
            return

        logger.info("stopping_frame_client")

        self.running = False
        self.shutdown_event.set()

        #close socket
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logger.warning("socket_close_error", error=str(e))
            finally:
                self.socket = None

        if self.receiver_task:
            try:
                await asyncio.wait_for(self.receiver_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("receiver_task_timeout_during_shutdown")
                self.receiver_task.cancel()

        self.state = ConnectionState.SHUTDOWN
        logger.info("frame_client_stopped")

    @async_retry(max_attempts=3, delay=1.0, exceptions=(CameraConnectionError, ))
    async def _connect(self) -> bool:
        """Connect to frame server. - TCP connection"""
        if self.state == ConnectionState.CONNECTED:
            return True

        self.state = ConnectionState.CONNECTING

        try:
            logger.info("connecting_to_frame_server", host=self.config.host, port=self.config.port)

            #socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.settimeout(self.config.timeout)

            #connect to frame server
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.socket.connect,
                (self.config.host, self.config.port)
            )

            #socket -> non-blocking
            # self.socket.setblocking(False)

            self.state = ConnectionState.CONNECTED
            self.reconnect_attempts = 0

            logger.info("connected_to_frame_server")
            return True

        except Exception as e:
            self.state = ConnectionState.ERROR
            if self.socket:
                self.socket.close()
                self.socket = None

            logger.error("canon_connection_failed", error=str(e))
            raise CameraConnectionError(f"Failed to connect to Canon FrameServer: {e}") from e

    async def _receiver_loop(self) -> None:
        """Receive frames from frame server."""

        data_buffer = bytearray()

        while self.running and not self.shutdown_event.is_set():
            try:
                #ensure connection
                if not await self._ensure_connected():
                    await asyncio.sleep(1.0)
                    continue

                #receive data
                try:
                    chunk = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._receive_data_chunk
                    )
                    if not chunk:
                        logger.warning("empty_data_received_connection_closed")
                        self.state = ConnectionState.ERROR
                        continue
                    data_buffer.extend(chunk)
                    self.bytes_received += len(chunk)
                except socket.timeout:
                    continue  # Normal timeout, retry
                except (ConnectionError, OSError) as e:
                    logger.warning("connection_error_during_receive", error=str(e))
                    self.state = ConnectionState.ERROR
                    continue

                #process complete frames in buffer
                while len(data_buffer) >= 4:
                    frame, bytes_consumed = await self._try_extract_frame(data_buffer)

                    if frame is not None:
                        data_buffer = data_buffer[bytes_consumed:]
                        await self._process_received_frame(frame)
                    else:
                        break
            except Exception as e:
                logger.error("receiver_loop_error", error=str(e))
                self.state = ConnectionState.ERROR
                await asyncio.sleep(1.0)

        logger.info("receiver_loop_stopped")

    def _receive_data_chunk(self) -> bytes:
        """Receive data chunks from the socket."""
        if not self.socket:
            raise ConnectionError("socket_not_connected")

        try:
            return self.socket.recv(4096)
        except socket.timeout:
            return b''
        except (ConnectionResetError, ConnectionAbortedError) as e:
            raise ConnectionError(f"Connection lost: {e}") from e

    async def _try_extract_frame(self, buffer: bytearray) -> tuple[Optional[np.ndarray], int]:
        """Try to extract frame from data buffer."""
        if len(buffer) < 4:
            return None, 0

        try:
            #read message length (first 4 bytes, little-endian)
            msg_size  = struct.unpack("<L", buffer[:4])[0]

            #validate msg
            if msg_size > 10 *1024 *1024:
                logger.frame("invalid_frame_size", size=msg_size)
                return None, 4 #skip invalid length header

            total_size = msg_size + 4 #header + payload

            if len(buffer) < total_size:
                return None, 0 #not enough data for complete frame

            #extract frame data
            frame_data = bytes(buffer[4:total_size])

            #decode JPEG frame
            frame = await self._decode_frame(frame_data)

            return frame, total_size
        except Exception as e:
            logger.error("frame_extraction_error", error=str(e))
            return None, 4 # skip problematic data

    async def _decode_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        """Decode frame data."""
        try:
            thread_manager = get_thread_manager()

            def _decode():
                img_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return frame

            frame = await thread_manager.run_io_task(_decode)

            if frame is None:
                logger.warning("frame_decode_failed")
                return None

            return frame

        except Exception as e:
            logger.error("decode_frame_error", error=str(e))
            return None


    async def _process_received_frame(self, frame: np.ndarray) -> None:
        """Process received frame."""
        current_time = time.time()

        #update performance metrics
        self.frames_received += 1
        self.last_frame_time = current_time
        current_fps = self.fps_calculator.update(current_time)

        #calculate frame latency
        latency_ms = 33.0 #approx for 30 fps

        #create frame metadata
        metadata = FrameMetadata(
            timestamp=current_time,
            frame_id=self.frames_received,
            size=frame.shape[:2],
            fps=current_fps,
            latency_ms=latency_ms
        )

        #store latest frame
        self.latest_frame = frame.copy()
        self.latest_metadata = metadata

        #add to queue
        try:
            self.frame_queue.put_nowait((frame, metadata))
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((frame, metadata))
            except queue.Empty:
                pass

        if self.frame_callback:
            try:
                self.frame_callback(frame, metadata)
            except Exception as e:
                logger.error("frame_callback_error", error=str(e))

        logger.debug(
            "frame_processed",
            frame_id=metadata.frame_id,
            fps=current_fps,
            size=metadata.size,
            queue_size=self.frame_queue.qsize
        )

    async def _ensure_connected(self) -> bool:
        """Check if connection is established, reconnect if needed."""
        if self.state == ConnectionState.CONNECTED:
            return True

        current_time = time.time()
        if (self.reconnect_attempts >= self.config.reconnect_attempts or
        current_time - self.last_reconnect_time < self.config.reconnect_delay):
            return False

        #reconnect
        self.last_reconnect_time = current_time
        self.reconnect_attempts += 1

        try:
            await self._connect()
            return True
        except CameraConnectionError as e:
            return False


    async def get_latest_frame(self, timeout: float = 0.1) -> Optional[tuple[np.ndarray, FrameMetadata]]:
        """Get latest frame asynchronously."""
        try:
            return await asyncio.wait_for(self.frame_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def get_latest_frame_sync(self) -> Optional[tuple[np.ndarray, FrameMetadata]]:
        """Get latest frame synchronously."""
        if self.latest_frame is not None and self.latest_metadata is not None:
            return self.latest_frame.copy(), self.latest_metadata
        return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        current_time = time.time()
        uptime = current_time - self.last_reconnect_time if self.last_reconnect_time > 0 else 0

        return {
            'state': self.state.value,
            'frames_received': self.frames_received,
            'bytes_received': self.bytes_received,
            'current_fps': self.fps_calculator.get_fps(),
            'reconnect_attempts': self.reconnect_attempts,
            'uptime_seconds': round(uptime, 1),
            'queue_size': self.frame_queue.qsize,
            'last_frame_age_ms': round((current_time - self.last_frame_time) * 1000, 1) if self.last_frame_time > 0 else None
        }

    @log_performance("frame_client_validation")
    async def validate_connection(self) -> Dict[str, Any]:
        """Validate frame client connection and performance"""
        validation_start = time.time()
        initial_frames = self.frames_received

        frames_collected = []
        timeout = 5.0

        start_time = time.time()
        while time.time() - start_time < timeout and len(frames_collected) < 10:
            frame_date = await self.get_latest_frame(timeout = 0.5)
            if frame_date:
                frames_collected.append(frame_date)
            await asyncio.sleep(0.1)

        validation_time = time.time() - validation_start
        frames_received_during_test = self.frames_received - initial_frames

        validation_result = {
            'connection_successful': self.state == ConnectionState.CONNECTED,
            'frames_received_during_test': frames_received_during_test,
            'frames_collected': len(frames_collected),
            'validation_time_seconds': round(validation_time, 2),
            'average_fps_during_test': round(frames_received_during_test / validation_time, 1),
            'performance_stats': self.get_performance_stats()
        }

        if frames_collected:
            frame, metadata = frames_collected[-1]
            validation_result.update({
                'latest_frame_size': frame.shape,
                'latest_frame_metadata': metadata.__dict__
            })

        logger.info("frame_client_validation_completed", **validation_result)
        return validation_result

# Convenience function
async def create_frame_client(
    frame_callback: Optional[Callable[[np.ndarray, FrameMetadata], None]] = None
) -> CanonFrameClient:
    """Create a frame client."""
    client = CanonFrameClient(frame_callback=frame_callback)
    await client.start()
    return client













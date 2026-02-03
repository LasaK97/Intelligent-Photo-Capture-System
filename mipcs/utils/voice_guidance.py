from pathlib import Path
from typing import List, Optional, Dict
import sys
import random
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_workflow_config, DictWrapper
from ..utils.logger import get_logger

logger = get_logger(__name__)


# Mapping from Photo Capture action_keys to VA audio_ids
# Format: "category.message_type" matching photo_capture.yaml in VA
ACTION_TO_AUDIO_ID: Dict[str, str] = {
    # Greeting
    "welcome": "greeting.session_start",

    # Status/Detection
    "single_person_detected": "status.person_detected",
    "couple_detected": "status.multiple_detected",
    "group_detected": "status.multiple_detected",
    "perfect_position": "status.position_perfect",

    # Positioning guidance
    "move_closer": "positioning.move_closer",
    "move_further": "positioning.move_back",
    "move_left": "positioning.move_left",
    "move_right": "positioning.move_right",
    "move_together": "positioning.move_together",
    "spread_out": "positioning.spread_apart",

    # Preparation
    "countdown_start": "preparation.ready",

    # Countdown (VA uses "three", "two", "one" not numbers)
    "countdown_3": "countdown.three",
    "countdown_2": "countdown.two",
    "countdown_1": "countdown.one",

    # Completion
    "capture_complete": "completion.success",

    # Errors
    "timeout_warning": "error.timeout",
}


class VoiceGuidanceMapper:
    """Maps action keys to audio_ids for pre-generated audio playback."""

    def __init__(self):
        workflow_dict = get_workflow_config()
        self.workflow_config = DictWrapper(workflow_dict)
        self.messages = self.workflow_config.voice_guidance.messages
        self.audio_id_map = ACTION_TO_AUDIO_ID

        logger.info("voice_guidance_mapper_initialized",
                    language=self.workflow_config.voice_guidance.language,
                    audio_ids_loaded=len(self.audio_id_map))

    def get_audio_id(self, action_key: str) -> Optional[str]:
        """Get audio_id for the given action key.

        Args:
            action_key: The action key (e.g., "move_closer")

        Returns:
            audio_id string (e.g., "positioning.move_closer") or None if not found
        """
        audio_id = self.audio_id_map.get(action_key)

        if audio_id is None:
            logger.warning("no_audio_id_for_action", action_key=action_key)
            return None

        logger.debug("audio_id_resolved", action_key=action_key, audio_id=audio_id)
        return audio_id

    def get_message(self, action_key: str) -> Optional[str]:
        """Get text message from action key (for fallback/logging)."""
        try:
            message = getattr(self.messages, action_key)

            if isinstance(message, list):
                return random.choice(message)

            return message
        except AttributeError:
            logger.warning("unknown_action_key", key=action_key)
            return None

    def get_messages_for_actions(self, action_keys: List[str]) -> List[Optional[str]]:
        """Get messages for multiple action keys."""
        messages = []

        for key in action_keys:
            msg = self.get_message(key)
            if msg:
                messages.append(msg)
        return messages


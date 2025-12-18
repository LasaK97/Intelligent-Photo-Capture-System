from pathlib import Path
from typing import List, Optional
import sys
import random
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_workflow_config, DictWrapper
from ..utils.logger import get_logger

logger = get_logger(__name__)

class VoiceGuidanceMapper:
    """maps action keys to the messages"""

    def __init__(self):
        workflow_dict = get_workflow_config()
        self.workflow_config = DictWrapper(workflow_dict)
        self.messages = self.workflow_config.voice_guidance.messages

        logger.info("voice_guidance_mapper_initialized. ", language=self.workflow_config.voice_guidance.language)

    def get_message(self, action_key: str) -> Optional[str]:
        """get message from action key"""

        try:
            message = getattr(self.messages, action_key)

            if isinstance(message, list):
                return random.choice(message)

            return message
        except AttributeError:
            logger.warning("unknown_action_key", key=action_key)
            return None

    def get_messages_for_actions(self, action_keys: List[str]) -> List[Optional[str]]:
        """get messages for multiple action keys"""
        messages = []

        for key in action_keys:
            msg = self.get_message(key)
            if msg:
                messages.append(msg)
        return messages


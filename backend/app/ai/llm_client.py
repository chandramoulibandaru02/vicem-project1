import logging

from app.core.config import get_settings


class LLMClient:
    def __init__(self) -> None:
        self.logger = logging.getLogger("ecm_ai_backend")
        self.settings = get_settings()

    async def initialize(self) -> None:
        self.logger.info("LLM client initialized for model: %s", self.settings.model_name)

    async def shutdown(self) -> None:
        self.logger.info("LLM client shutdown")

    async def generate_answer(self, prompt: str) -> str:
        self.logger.info("Generating prototype answer for prompt length: %s", len(prompt))
        return f"Prototype response using model {self.settings.model_name}: {prompt[:120]}"

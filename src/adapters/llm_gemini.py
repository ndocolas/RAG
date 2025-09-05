import time
import logging
from google import genai
from google.genai import types
from typing import Generator
from src.core.config import settings

log = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.GEMINI_MODEL

    def _backoff(self, attempt : int): time.sleep(min(2**attempt, 8))

    def generate(self, prompt : str, system : str | None = None, tries : int = 3) -> str:
        cfg = types.GenerateContentConfig(
            system_instruction=system or "",
        )
        for i in range(tries):
            try:
                r = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
                return r.text or ""
            except Exception as e:
                log.warning(f"Gemini generate failed ({i+1}/{tries}): {e}")
                if i == tries - 1:
                    raise e
                self._backoff(i)
        return ""

    def stream(self, prompt : str, system : str | None = None, tries : int = 3) -> Generator[str, None, None]:
        cfg = types.GenerateContentConfig(system_instruction=system or "")
        for i in range(tries):
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model, contents=prompt, config=cfg
                ):
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                log.warning(f"Gemini stream failed ({i+1}/{tries}): {e}")
                if i == tries - 1:
                    raise e
                self._backoff(i)

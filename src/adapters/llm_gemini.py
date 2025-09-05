from google import genai
from src.models.v1.settings import settings
from src.models.v1.errors import GeminiException
import logging


class GeminiClient:
    logger = logging.getLogger(__name__)
    
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.GEMINI_MODEL

    def invoke(self, prompt: str) -> str:
        try:
            self.logger.debug("\nRecieve LLM request")

            llm_response = self.client.models.generate_content(model=self.model, contents=prompt)
            
            self.logger.debug("\nGenerated LLM response: %s", llm_response)
            
            if not llm_response.text:
                self.logger.error("\nAn error occurred while generating response: %s", llm_response)
                raise GeminiException
            
            return llm_response.text
        except Exception as e:
                self.logger.error("\nAn error occurred while generating response: %s", e)
                raise e

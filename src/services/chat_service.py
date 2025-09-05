import logging
from src.models.v1.dto import AIResponse
from src.services.rag import RAGPipeline
from src.adapters.llm_gemini import GeminiClient
from src.adapters.system_prompt import build_prompt

class ChatService:
    logger = logging.getLogger(__name__)

    def __init__(self, llm: GeminiClient):
        self.llm = llm
        self.rag = RAGPipeline()

    def chat(self, session_id: str, user_input: str) -> AIResponse:
        try:
            self.logger.info("\nReceived chat request | session_id=%s | user_input=%s", session_id, user_input)

            related_chunk = self.rag.get_chunk(session_id=session_id, question=user_input)
            self.logger.info("\nRetrieved chunk: %.80s...", related_chunk)

            prompt = build_prompt(user_input=user_input, chunk=related_chunk)
            self.logger.info("\nBuilt prompt: %.120s...", prompt)

            llm_response = self.llm.invoke(prompt)
            self.logger.info("\nGemini returned response (len=%d)", len(llm_response))

            return AIResponse(
                session_id=session_id,
                user_input=user_input,
                response=llm_response,
            )
        except Exception as e:
            self.logger.error("\nChat failed: %s", e)
            raise e
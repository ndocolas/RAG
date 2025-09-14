"""Service to handle chat interactions using a LLM."""

import logging

from src.backend.services.chat_service.models.models import ChatOutput
from src.backend.services.chat_service.llm_builder import LLMBuilder
from src.backend.services.chat_service.models.models import AIChatOutput
from src.backend.services.chat_service.system_prompt import PROMPT, PROMPT_VARIABLES
from langchain.prompts import PromptTemplate


class ChatService:
    """Service to handle chat interactions using a LLM."""

    logger = logging.getLogger(__name__)

    def __init__(self):
        """Initializes the ChatService with a LLM instance."""
        self.llm = LLMBuilder.build_llm()

    async def chat(self, session_id: str, user_input: str) -> ChatOutput:
        """Handles a chat interaction.

        Args:
            session_id (str): The session ID for the chat.
            user_input (str): The user's input question.
        """
        try:
            self.logger.info(
                "\nReceived chat request. \nsession_id: %s\nuser_input: %s",
                session_id,
                user_input,
            )

            llm_with_structured_output = self.llm.with_structured_output(schema=AIChatOutput)

            prompt_template = PromptTemplate(
                input_variables=PROMPT_VARIABLES,
                template=PROMPT,
            )
            chain = prompt_template | llm_with_structured_output

            # get most related chunk
            # related_chunk = self.rag.get_chunk(
            #     session_id=session_id, question=user_input
            # )
            # self.logger.info("\nRetrieved c. hunk: %.80s...", related_chunk)

            response = await chain.ainvoke({"user_input": user_input, "chunk": "related_chunk"})

            response_model = AIChatOutput.model_validate(response)

            return ChatOutput(
                user_input=user_input,
                session_id=session_id,
                response_model=response_model,
            )
        except Exception as e:
            self.logger.error("\nChat failed: %s", e)
            raise e

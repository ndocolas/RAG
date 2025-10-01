"""Chat service module to handle interactions with a Large Language Model (LLM) and manage chat sessions."""

import logging
from langchain.prompts import PromptTemplate

from src.backend.services.chat_service.models.models import ChatOutput
from src.backend.services.chat_service.llm_builder import LLMBuilder
from src.backend.services.chat_service.models.models import AIChatOutput
from src.backend.services.chat_service.system_prompt import PROMPT, PROMPT_VARIABLES
from src.backend.services.vector_service.vector_service import RetrievalService


class ChatService:
    """Service to handle chat interactions using a Large Language Model (LLM)."""

    logger = logging.getLogger(__name__)

    def __init__(self):
        """Initialize the ChatService with an LLM instance and retrieval service.

        Inputs:
        None

        Returns:
        None: Initializes internal components for LLM interaction and document retrieval
        """
        self.llm = LLMBuilder.build_llm()
        self.retrieval = RetrievalService()

    async def chat(self, session_id: str, user_input: str) -> ChatOutput:
        """Handle a chat request, retrieve context, run the LLM chain, and return the structured response.

        Inputs:
        session_id: Unique session identifier used to retrieve the correct FAISS index
        user_input: The text query provided by the user

        Returns:
        ChatOutput: The final structured chat response containing user input, session id, and the model output
        """
        try:
            self.logger.info(
                "\nReceived chat request. \nsession_id: %s\nuser_input: %s",
                session_id,
                user_input,
            )

            context, _docs = self.retrieval.top_context(session_id, user_input, k=1)
            self.logger.info(
                "\nRetrieved context (%.80s...)", context.replace("\n", " ")
            )

            llm_with_structured_output = self.llm.with_structured_output(
                schema=AIChatOutput
            )
            prompt_template = PromptTemplate(
                input_variables=PROMPT_VARIABLES, template=PROMPT
            )
            chain = prompt_template | llm_with_structured_output

            response = await chain.ainvoke({"user_input": user_input, "chunk": context})
            response_model = AIChatOutput.model_validate(response)

            return ChatOutput(
                user_input=user_input,
                session_id=session_id,
                response_model=response_model,
            )
        except Exception as e:
            self.logger.error("\nChat failed: %s", e)
            raise e

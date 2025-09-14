"""Module for creating a LLM builder."""

from langchain_google_genai import ChatGoogleGenerativeAI
from src.backend.secrets.settings import settings


class LLMBuilder:
    """Class to build a LLM (Large Language Model) instance."""

    def __init__(self):
        """This class is not intended to be instantiated directly."""
        raise TypeError(
            "This class is not intended to be instantiated directly. "
            "Use its static method directly from the class itself."
        )

    @staticmethod
    def build_llm() -> ChatGoogleGenerativeAI:
        """Builds a LLM instance.

        Returns:
            llm: An instance of the LLM class.
        """
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            api_key=settings.GEMINI_API_KEY,
        )

        return llm

"""Module for creating an LLM builder."""

from langchain_google_genai import ChatGoogleGenerativeAI
from src.backend.secrets.settings import settings


class LLMBuilder:
    """Class to build a Large Language Model (LLM) instance."""

    def __init__(self):
        """
        Prevent direct instantiation of this class.

        Inputs:
        None

        Returns:
        Raises a TypeError to enforce usage of the static method
        """
        raise TypeError(
            "This class is not intended to be instantiated directly. "
            "Use its static method directly from the class itself."
        )

    @staticmethod
    def build_llm() -> ChatGoogleGenerativeAI:
        """
        Build and return a configured LLM instance.

        Inputs:
        None

        Returns:
        ChatGoogleGenerativeAI: An instance of the LLM configured with the model and API key from settings
        """
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            api_key=settings.GEMINI_API_KEY,
        )

        return llm

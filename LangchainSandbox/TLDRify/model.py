from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class ChatModel:
    """Initializes and configures the active Google Gemini language model."""

    def __init__(
        self, model: str = "gemini-3.1-pro-preview", temperature: float = 0.0
    ) -> None:
        """
        Initialize the language model configuration.

        Args:
            model (str): Google Gemini AI model string identifier.
            temperature (float): Model temperature, where lower limits randomness.
        """
        try:
            self.model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
                model=model, temperature=temperature
            )

        except Exception as e:
            print(
                f"Error configuring ChatModel engine API references natively: {e}"
            )

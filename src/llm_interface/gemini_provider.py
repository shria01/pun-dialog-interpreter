import os
import time
from google import genai
from google.genai import errors, types
from .interface import LLMInterface

class GeminiProvider(LLMInterface):

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30_000),
        )
        self.validation_model = "gemini-3.5-flash-lite"
        self.chat_model = "gemini-2.5-flash"

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        code = getattr(error, "code", None)
        error_name = type(error).__name__.lower()
        return (
            isinstance(error, errors.ServerError)
            or code in {408, 429}
            or "timeout" in error_name
            or "connection" in error_name
        )

    def _generate_content(self, model: str, contents):
        for attempt in range(2):
            try:
                return self.client.models.generate_content(model=model, contents=contents)
            except Exception as error:
                retryable = self._is_transient(error)
                print(
                    "gemini_request_error "
                    f"model={model} type={type(error).__name__} "
                    f"code={getattr(error, 'code', None)} retry={retryable and attempt == 0}",
                    flush=True,
                )
                if not retryable or attempt == 1:
                    raise
                time.sleep(0.75)

    def generate(self, prompt: str) -> str:
        response = self._generate_content(self.validation_model, prompt)
        return response.text.strip()

    def chat(self, messages: list[dict]) -> str:
        # Convert from standard format {"role", "content"} → Gemini format
        gemini_messages = []
        for message in messages:
            if message["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": message["content"]}]})
            elif message["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": message["content"]}]})
            else:
                raise ValueError(f"Unknown role: {message['role']}")

        response = self._generate_content(self.chat_model, gemini_messages)
        return response.text.strip()

    def get_model_info(self) -> dict:
        return {
            "provider": "gemini",
            "validation_model": self.validation_model,
            "chat_model": self.chat_model,
        }

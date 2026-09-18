import os
import time
from openai import OpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from .interface import LLMInterface

class OpenAIProvider(LLMInterface):

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        # Retry transient failures in _create so the request count and delay are
        # explicit instead of multiplying SDK retries by application retries.
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
        self.model = "gpt-4o-mini"

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        return isinstance(
            error,
            (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError),
        )

    def _create(self, **kwargs):
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI response had no content (possibly filtered)")
                return content.strip()
            except Exception as error:
                retryable = self._is_transient(error)
                print(
                    "openai_request_error "
                    f"type={type(error).__name__} retry={retryable and attempt == 0}",
                    flush=True,
                )
                if not retryable or attempt == 1:
                    raise
                time.sleep(0.75)

    def generate(self, prompt: str) -> str:
        return self._create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

    def chat(self, messages: list[dict]) -> str:
        return self._create(model=self.model, messages=messages)

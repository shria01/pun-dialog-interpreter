import os
from google import genai
from .interface import LLMInterface

class GeminiProvider(LLMInterface):

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
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
            
        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_messages
        )
        return response.text.strip()
    
    def get_model_info(self) -> dict:
        return {
            "provider": "gemini",
            "model": self.model
        }
    
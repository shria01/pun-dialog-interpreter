import os
from google import genai
from .interface import LLMInterface

class GeminiProvider(LLMInterface):
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        self.model = "gemini-2.5-flash"

    def generate_content(self, prompt):
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
                gemini_messages.append({"author": "user", "parts": [{"text": message["content"]}]})
            elif message["role"] == "assistant":
                gemini_messages.append({"author": "assistant", "parts": [{"text": message["content"]}]})
            else:
                raise ValueError(f"Unknown role: {message['role']}")
            
        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_messages
        )
        return response.text.strip()
    
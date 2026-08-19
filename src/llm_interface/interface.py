from abc import ABC, abstractmethod

class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Single-shot prompt → string response."""
        pass

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Multi-turn conversation. messages = [{"role": ..., "content": ...}]"""
        pass

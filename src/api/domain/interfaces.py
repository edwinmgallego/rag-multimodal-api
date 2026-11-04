from abc import ABC, abstractmethod

class ILLMProvider(ABC):
    @abstractmethod
    def generate_answer(self, context: str, query: str) -> str:
        """Genera una respuesta basada estrictamente en el contexto."""
        pass
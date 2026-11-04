import os
from openai import OpenAI
from src.domain.interfaces import ILLMProvider

class OpenAIService(ILLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate_answer(self, context: str, query: str) -> str:
        """Genera una respuesta aplicando Prompt Engineering defensivo."""
        
        system_prompt = (
            "Eres un asistente técnico experto. Tu tarea es responder a la pregunta del usuario "
            "basándote ÚNICA y EXCLUSIVAMENTE en el contexto técnico proporcionado.\n\n"
            "REGLAS ESTRICTAS:\n"
            "1. Si el contexto proporcionado no contiene la respuesta, debes responder EXACTAMENTE: "
            "'No tengo información suficiente en los manuales para responder a esta pregunta.'\n"
            "2. No inventes información, no asumas datos que no estén en el texto.\n"
            "3. Responde en formato Markdown, usando listas y negritas cuando sea necesario para mayor claridad."
        )
        
        user_prompt = f"Contexto técnico recuperado:\n{context}\n\nPregunta del usuario: {query}"
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo", # O gpt-4o-mini según tu presupuesto
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # Temperatura 0 para evitar creatividad y alucinaciones
        )
        
        return response.choices[0].message.content
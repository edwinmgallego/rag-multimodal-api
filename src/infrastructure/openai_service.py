import os
from openai import OpenAI, RateLimitError, APIConnectionError, APIError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from src.domain.interfaces import ILLMProvider

class OpenAIService(ILLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    # Configuramos Tenacity: Reintenta hasta 4 veces, esperando exponencialmente entre 2 y 10 segundos
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
        reraise=True
    )
    def generate_answer(self, context: str, query: str) -> str:
        """Genera una respuesta aplicando Prompt Engineering defensivo y Resiliencia."""
        
       # system_prompt = (
       #     "Eres un asistente técnico experto. Tu tarea es responder a la pregunta del usuario "
       #     "basándote ÚNICA y EXCLUSIVAMENTE en el contexto técnico proporcionado.\n\n"
       #     "REGLAS ESTRICTAS:\n"
       #     "1. Si el contexto proporcionado no contiene la respuesta, debes responder EXACTAMENTE: "
       #     "'No tengo información suficiente en los manuales para responder a esta pregunta.'\n"
       #     "2. No inventes información, no asumas datos que no estén en el texto.\n"
       #     "3. Responde en formato Markdown, usando listas y negritas cuando sea necesario."
      #  )
        system_prompt = (
        "Eres un Dungeon Master experto y un asistente técnico. Tu tarea es responder a la pregunta del usuario "
        "basándote en el contexto técnico proporcionado.\n\n"
        "REGLAS:\n"
        "1. Utiliza toda la información disponible en el contexto para construir una respuesta completa.\n"
        "2. Si el contexto menciona partes de la respuesta, úsalas y dedúcelo lógicamente sin inventar reglas nuevas.\n"
        "3. Responde en formato Markdown, usando listas y negritas para que sea fácil de leer."
    )
        
        user_prompt = f"Contexto técnico recuperado:\n{context}\n\nPregunta del usuario: {query}"
        
        try:
            print("[OpenAI] Solicitando inferencia al LLM...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[OpenAI Error] Fallo detectado: {str(e)}. Tenacity gestionará el reintento si aplica.")
            raise e  # Lanzamos el error intencionalmente para que el decorador @retry lo intercepte
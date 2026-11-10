import pytest
from fastapi.testclient import TestClient
from src.api.main import app, get_llm_provider, get_vector_db
from src.domain.interfaces import ILLMProvider

# 1. Crear los Mocks para aislar la infraestructura externa

class MockLLMProvider(ILLMProvider):
    def generate_answer(self, context: str, query: str) -> str:
        # Simulamos que el LLM responde correctamente sin gastar tokens de OpenAI
        return "Respuesta simulada basada en el contexto."

class MockVectorDB:
    def search_context(self, query: str, limit: int = 3):
        # Simulamos lo que devolvería Qdrant al encontrar fragmentos relevantes
        return [
            {
                "score": 0.95,
                "text": "La pieza X del motor debe ensamblarse primero.",
                "page": 5,
                "associated_images": ["/static/images/mock_img.png"]
            }
        ]

# 2. Configurar el cliente de pruebas

client = TestClient(app)

# 3. Escribir la prueba unitaria

def test_chat_endpoint_with_mocks():
    """
    Verifica que el endpoint /chat integre correctamente la búsqueda híbrida 
    y el LLM, utilizando mocks para evitar llamadas reales a la red.
    """
    # Sobrescribir las dependencias reales con nuestros mocks
    app.dependency_overrides[get_llm_provider] = lambda: MockLLMProvider()
    app.dependency_overrides[get_vector_db] = lambda: MockVectorDB()
    
    # Simular la petición del usuario
    payload = {"query": "¿Qué pieza se ensambla primero?"}
    response = client.post("/chat", json=payload)
    
    # Validar el código de estado
    assert response.status_code == 200
    
    # Validar la estructura y contenido de la respuesta
    data = response.json()
    assert data["answer"] == "Respuesta simulada basada en el contexto."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["page"] == 5
    assert data["sources"][0]["image_url"] == "/static/images/mock_img.png"
    
    # Limpiar las dependencias después de la prueba
    app.dependency_overrides.clear()
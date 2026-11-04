import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI

class QdrantRepository:
    def __init__(self, collection_name: str = "document_chunks"):
        self.collection_name = collection_name
        self.qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Crea la colección en Qdrant si no existe, configurada para vectores de OpenAI (1536 dimensiones)."""
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def generate_embeddings_and_store(self, chunks: List[Dict[str, Any]], job_id: str):
        """Convierte los fragmentos de texto en vectores y los guarda con sus metadatos multimodales."""
        points = []
        
        for chunk in chunks:
            # 1. Llamada a OpenAI para obtener el vector del fragmento de texto
            response = self.openai.embeddings.create(
                input=chunk["text"],
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            
            # 2. Estructurar el punto para Qdrant, inyectando las URLs de las imágenes como metadatos
            points.append(
                PointStruct(
                    id=chunk["chunk_id"],  # Ej: p1_c0
                    vector=embedding,
                    payload={
                        "job_id": job_id,
                        "text": chunk["text"],
                        "page": chunk["metadata"]["page"],
                        "associated_images": chunk["metadata"]["associated_images"]
                    }
                )
            )
            
        # 3. Inserción masiva (upsert) en la base de datos
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
    def search_context(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        CU-04: Recupera el contexto semántico más relevante para una pregunta.
        Retorna el texto y los metadatos visuales (imágenes) asociados.
        """
        # 1. Convertir la pregunta del usuario a un vector numérico (Embedding)
        response = self.openai.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_vector = response.data[0].embedding
        
        # 2. Ejecutar la búsqueda de similitud en Qdrant
        # (Para un híbrido puro con BM25, se configuraría un Sparse Vector adicional)
        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True # Crucial para recuperar el texto y las rutas de las imágenes
        )
        
        # 3. Estructurar los resultados para inyectarlos al LLM y al Frontend
        context_data = []
        for point in search_results:
            payload = point.payload
            context_data.append({
                "score": point.score,
                "text": payload.get("text", ""),
                "page": payload.get("page", "Desconocida"),
                "associated_images": payload.get("associated_images", [])
            })
            
        return context_data
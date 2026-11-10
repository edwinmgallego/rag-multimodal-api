import os
import uuid
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
        """Crea la colección en Qdrant si no existe."""
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def generate_embeddings_and_store(self, chunks: List[Dict[str, Any]], job_id: str):
        """Convierte los fragmentos de texto en vectores y los guarda en lotes."""
        points = []
        
        for chunk in chunks:
            # 1. Llamada a OpenAI para obtener el vector del fragmento de texto
            response = self.openai.embeddings.create(
                input=chunk["text"],
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            
            # 2. Generar un UUID válido para Qdrant usando el chunk_id original
            valid_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))

            points.append(
                PointStruct(
                    id=valid_uuid,
                    vector=embedding,
                    payload={
                        "job_id": job_id,
                        "original_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "page": chunk["metadata"]["page"],
                        "associated_images": chunk["metadata"]["associated_images"]
                    }
                )
            )
            
        # 3. Inserción en lotes (Batches) para evitar el error de 32MB
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"[JOB {job_id}] Lote insertado en Qdrant: {i} a {i + len(batch)}")

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
        response_qdrant = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector, 
            limit=limit,
            with_payload=True
        )
        search_results = response_qdrant.points
        
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
from celery import Celery
import os
from src.infrastructure.pdf_extractor import PyMuPDFExtractor
from src.infrastructure.chunker import SemanticChunker
from src.infrastructure.qdrant_repository import QdrantRepository

# Configuración conectada al contenedor Redis definido en docker-compose
celery_app = Celery(
    "rag_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
)

@celery_app.task(name="process_document_task")
def process_document(job_id: str, file_path: str):
    """CU-02: Procesamiento Asíncrono y Extracción Multimodal"""
    print(f"[JOB {job_id}] Iniciando procesamiento de {file_path}")
    
    try:
        # 1. Extracción Multimodal
        extractor = PyMuPDFExtractor(output_image_dir="./data/images")
        extracted_data = extractor.extract(file_path, job_id)
        
        # 2. Chunking Semántico
        chunker = SemanticChunker(chunk_size=800, chunk_overlap=150)
        chunks = chunker.split(extracted_data)
        
        print(f"[JOB {job_id}] Éxito: {len(chunks)} fragmentos semánticos creados.")
        
        # 3. Vectorización y guardado en Qdrant
        print(f"[JOB {job_id}] Generando embeddings y guardando en Vector DB...")
        qdrant_repo = QdrantRepository(collection_name="document_chunks")
        qdrant_repo.generate_embeddings_and_store(chunks, job_id)
        
        print(f"[JOB {job_id}] Proceso completado exitosamente.")
        
        return {"job_id": job_id, "status": "Completed", "chunks_created": len(chunks)}
        
    except Exception as e:
        print(f"[JOB {job_id}] Error crítico: {str(e)}")
        return {"job_id": job_id, "status": "Failed", "error": str(e)}
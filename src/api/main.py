from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import shutil
import os

from src.infrastructure.celery_worker import process_document
from src.domain.interfaces import ILLMProvider
from src.infrastructure.openai_service import OpenAIService
from src.infrastructure.qdrant_repository import QdrantRepository

# --- ESQUEMAS PYDANTIC ---
class UploadResponse(BaseModel):
    job_id: str = Field(...)
    filename: str
    status: str = Field(default="Pending")

class StatusResponse(BaseModel):
    job_id: str
    status: str

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=5)

class SourceMetadata(BaseModel):
    page: int
    image_url: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata] = Field(default_factory=list)

app = FastAPI(title="RAG Multimodal API", version="1.0.0")

# --- MOCK PARA EVITAR ERROR 429 DE OPENAI ---
class MockLLMProvider(ILLMProvider):
    def generate_answer(self, context: str, query: str) -> str:
        return "¡Éxito! Esta es una respuesta simulada. Tu motor RAG recuperó los fragmentos del PDF correctamente desde Qdrant. La conexión con OpenAI está desactivada temporalmente."

app = FastAPI(title="RAG Multimodal API", version="1.0.0")

# --- INYECCIÓN DE DEPENDENCIAS ---
def get_llm_provider() -> ILLMProvider:
    return OpenAIService() # <-- Línea original comentada
    ##return MockLLMProvider() # <-- Inyectamos el Mock

def get_vector_db() -> QdrantRepository:
    return QdrantRepository(collection_name="document_chunks")

# --- ENDPOINTS ---
@app.post("/upload", response_model=UploadResponse, tags=["Ingesta"])
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten PDFs.")
    
    job_id = str(uuid.uuid4())
    os.makedirs("./data", exist_ok=True)
    file_path = f"./data/{job_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    process_document.delay(job_id, file_path)
    return UploadResponse(job_id=job_id, filename=file.filename)

@app.get("/status/{job_id}", response_model=StatusResponse, tags=["Ingesta"])
def get_status(job_id: str):
    return StatusResponse(job_id=job_id, status="Processing")


@app.post("/chat", response_model=ChatResponse, tags=["Motor RAG"])
def chat_with_rag(
    request: ChatRequest, 
    llm: ILLMProvider = Depends(get_llm_provider),
    vector_db: QdrantRepository = Depends(get_vector_db)
):
    """CU-04: Consulta RAG (Chat Híbrido Multimodal)"""
    
    # 1. Recuperar contexto REAL de Qdrant (Quitamos el Mock)
    search_results = vector_db.search_context(query=request.query, limit=3)
    
    if not search_results:
        return ChatResponse(
            answer="No tengo información suficiente en los manuales procesados para responder a esta pregunta.",
            sources=[]
        )

    # 2. Consolidar el texto y extraer las imágenes para las fuentes
    context_texts = []
    sources = []
    
    for result in search_results:
        context_texts.append(result["text"])
        # Tomar la primera imagen asociada a este fragmento de texto si existe
        img_url = result["associated_images"][0] if result.get("associated_images") else None
        
        sources.append(
            SourceMetadata(page=result["page"], image_url=img_url)
        )
        
    merged_context = "\n\n---\n\n".join(context_texts)
    
    # 3. Generar la respuesta usando OpenAI real
    answer = llm.generate_answer(context=merged_context, query=request.query)
    
    return ChatResponse(answer=answer, sources=sources)
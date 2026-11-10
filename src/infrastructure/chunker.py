from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Configuración para respetar la estructura del documento técnico
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split(self, extracted_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Divide el texto de las páginas en fragmentos (chunks) y preserva 
        los metadatos clave, incluyendo las imágenes de la página.
        """
        document_chunks = []
        
        for page_data in extracted_pages:
            page_text = page_data["text"]
            page_num = page_data["page"]
            images = page_data["images"]
            
            # Dividir el texto respetando los separadores lógicos
            text_chunks = self.splitter.split_text(page_text)
            
            for index, chunk_text in enumerate(text_chunks):
                # Extraemos solo las URLs de las imágenes para guardarlas como metadatos en Qdrant
                image_urls = [img["url"] for img in images]
                
                document_chunks.append({
                    "chunk_id": f"p{page_num}_c{index}",
                    "text": chunk_text,
                    "metadata": {
                        "page": page_num,
                        "associated_images": image_urls # Clave para el contexto visual
                    }
                })
                
        return document_chunks
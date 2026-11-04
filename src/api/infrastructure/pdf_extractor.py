import fitz  # PyMuPDF
import os
from typing import List, Dict, Any

class PyMuPDFExtractor:
    def __init__(self, output_image_dir: str = "./data/images"):
        self.output_image_dir = output_image_dir
        os.makedirs(self.output_image_dir, exist_ok=True)

    def extract(self, file_path: str, job_id: str) -> List[Dict[str, Any]]:
        """
        Procesa el PDF y extrae texto e imágenes por página.
        Retorna una lista de diccionarios con el contenido multimodal.
        """
        doc = fitz.open(file_path)
        extracted_pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 1. Extraer Texto Estructurado
            text = page.get_text("text").strip()
            
            # 2. Buscar y Extraer Imágenes
            images_metadata = []
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0] # Identificador único de la imagen en el PDF
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Nombrar y guardar la imagen físicamente asociándola al job_id
                image_filename = f"{job_id}_p{page_num}_i{img_index}.{image_ext}"
                image_filepath = os.path.join(self.output_image_dir, image_filename)
                
                with open(image_filepath, "wb") as f:
                    f.write(image_bytes)
                
                # 3. Obtener Coordenadas Espaciales (Bounding Box)
                # get_image_rects devuelve dónde se ubica la imagen en la página
                rects = page.get_image_rects(xref)
                bbox = None
                if rects:
                    # Guardamos [x0, y0, x1, y1] (izquierda, arriba, derecha, abajo)
                    rect = rects[0]
                    bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                
                images_metadata.append({
                    "url": f"/static/images/{image_filename}", # Ruta para exponer en FastAPI
                    "local_path": image_filepath,
                    "bbox": bbox
                })
            
            extracted_pages.append({
                "page": page_num + 1,
                "text": text,
                "images": images_metadata
            })
            
        doc.close()
        return extracted_pages
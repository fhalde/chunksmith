import os
import base64
import fitz
import webview
from typing import List, Dict, Any
from .chunkers.basic import BasicWordChunker, SentenceChunker
from .chunkers.semantic import SemanticChunker
from .chunkers.topic import TopicChunker

class Api:
    def __init__(self):
        self._window = None
        # Initialize chunkers
        basic_chunker = BasicWordChunker()
        sentence_chunker = SentenceChunker()
        semantic_chunker = SemanticChunker()
        topic_chunker = TopicChunker()

        # Use the dynamic name property for the key
        self.chunkers = {
            basic_chunker.name: basic_chunker,
            sentence_chunker.name: sentence_chunker,
            semantic_chunker.name: semantic_chunker,
            topic_chunker.name: topic_chunker
        }

    def set_window(self, window):
        self._window = window

    def select_pdf(self) -> str:
        """Open a file dialog to select a PDF."""
        file_types = ('PDF Files (*.pdf)', 'All files (*.*)')
        result = self._window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types)
        if result:
            return result[0]
        return None

    def get_algorithms(self) -> List[Dict[str, str]]:
        """Return a list of available chunking algorithms."""
        return [
            {"name": c.name, "description": c.description}
            for c in self.chunkers.values()
        ]

    def process_pdf(self, pdf_path: str, algorithm_name: str) -> Dict[str, Any]:
        """Process the PDF with the selected algorithm."""
        if not os.path.exists(pdf_path):
            return {"error": "File not found"}

        chunker = self.chunkers.get(algorithm_name)
        if not chunker:
            return {"error": f"Algorithm '{algorithm_name}' not found. Available: {list(self.chunkers.keys())}"}

        try:
            # Get document info for rendering setup on frontend
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            # Get dimensions of the first page (assuming uniform, but frontend can handle per page)
            pages_info = []
            for i in range(page_count):
                page = doc[i]
                rect = page.rect
                pages_info.append({
                    "page": i,
                    "width": rect.width,
                    "height": rect.height
                })

            # Run chunking
            chunks = chunker.chunk(pdf_path)

            # Serialize chunks
            serialized_chunks = []
            for c in chunks:
                serialized_chunks.append({
                    "id": c.id,
                    "text": c.text,
                    "bboxes": [
                        {"page": b.page, "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1}
                        for b in c.bboxes
                    ],
                    "metadata": c.metadata
                })

            return {
                "page_count": page_count,
                "pages": pages_info,
                "chunks": serialized_chunks
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def get_page_image(self, pdf_path: str, page_num: int, scale: float = 1.5) -> str:
        """Render a PDF page to a base64 image."""
        try:
            doc = fitz.open(pdf_path)
            if page_num < 0 or page_num >= len(doc):
                return None

            page = doc[page_num]
            # Zoom factor
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)

            # Convert to png in memory
            img_data = pix.tobytes("png")
            base64_img = base64.b64encode(img_data).decode("utf-8")

            return f"data:image/png;base64,{base64_img}"
        except Exception as e:
            print(f"Error rendering page: {e}")
            return None

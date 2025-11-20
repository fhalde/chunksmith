import fitz
import uuid
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseChunker, Chunk, BoundingBox

class SemanticChunker(BaseChunker):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None

    @property
    def name(self) -> str:
        return "Semantic Chunker"

    @property
    def description(self) -> str:
        return f"Expands chunks based on semantic similarity (Model: {self.model_name}, Threshold: {self.threshold})."

    @property
    def model(self):
        if self._model is None:
            # Lazy load the model
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _extract_sentences(self, doc: fitz.Document) -> List[dict]:
        """Extract sentences with their bounding boxes from the PDF."""
        sentences = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue

                current_sent_text = ""
                current_sent_bboxes = []

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        bbox = span["bbox"]

                        current_sent_text += text + " "
                        current_sent_bboxes.append(BoundingBox(
                            page=page_num,
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3]
                        ))

                        # Naive sentence splitting
                        if text.strip().endswith((".", "!", "?")):
                            sentences.append({
                                "text": current_sent_text.strip(),
                                "bboxes": current_sent_bboxes
                            })
                            current_sent_text = ""
                            current_sent_bboxes = []

                # Capture any remaining text in the block as a sentence
                if current_sent_text.strip():
                    sentences.append({
                        "text": current_sent_text.strip(),
                        "bboxes": current_sent_bboxes
                    })

        return sentences

    def chunk(self, pdf_path: str) -> List[Chunk]:
        doc = fitz.open(pdf_path)

        # 1. Extract sentences
        sentences_data = self._extract_sentences(doc)
        if not sentences_data:
            return []

        # 2. Compute embeddings
        texts = [s["text"] for s in sentences_data]
        print(f"Generating embeddings for {len(texts)} sentences...")
        embeddings = self.model.encode(texts)

        chunks = []
        current_chunk_indices = [0]

        # 3. Group sentences
        for i in range(1, len(sentences_data)):
            current_embedding = np.mean(embeddings[current_chunk_indices], axis=0).reshape(1, -1)
            next_embedding = embeddings[i].reshape(1, -1)

            # Calculate similarity
            similarity = cosine_similarity(current_embedding, next_embedding)[0][0]

            if similarity >= self.threshold:
                # Expand chunk
                current_chunk_indices.append(i)
            else:
                # Depreciated similarity, finalize current chunk
                self._create_chunk_from_indices(chunks, sentences_data, current_chunk_indices)
                # Start new chunk
                current_chunk_indices = [i]

        # Finalize last chunk
        if current_chunk_indices:
            self._create_chunk_from_indices(chunks, sentences_data, current_chunk_indices)

        return chunks

    def _create_chunk_from_indices(self, chunks_list, sentences_data, indices):
        combined_text = " ".join([sentences_data[i]["text"] for i in indices])
        combined_bboxes = []
        for i in indices:
            combined_bboxes.extend(sentences_data[i]["bboxes"])

        chunk = Chunk(
            id=str(uuid.uuid4()),
            text=combined_text,
            bboxes=combined_bboxes,
            metadata={"sentence_count": len(indices)}
        )
        chunks_list.append(chunk)


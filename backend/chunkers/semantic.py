import fitz
import uuid
import numpy as np
from typing import List, Tuple, Dict
from .base import BaseChunker, Chunk, BoundingBox

class SemanticChunker(BaseChunker):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", percentile_threshold: float = 90.0, window_size: int = 1):
        """
        Args:
            model_name: HuggingFace model name.
            percentile_threshold: The percentile of cosine distances to use as a split threshold.
                                  Higher = fewer chunks (only splits at the most distinct transitions).
            window_size: Number of sentences to look ahead/behind for context when embedding.
        """
        self.model_name = model_name
        self.percentile_threshold = percentile_threshold
        self.window_size = window_size
        self._model = None

    @property
    def name(self) -> str:
        return "Semantic Chunker (Percentile)"

    @property
    def description(self) -> str:
        return f"Splits text at the top {100-self.percentile_threshold}% of semantic transitions. Groups related bullets."

    @property
    def model(self):
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            # Lazy import to speed up app startup
            from sentence_transformers import SentenceTransformer
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

                        # Handle basic whitespace to avoid words gluing together
                        if current_sent_text and not current_sent_text.endswith(" "):
                             current_sent_text += " "

                        current_sent_text += text
                        current_sent_bboxes.append(BoundingBox(
                            page=page_num,
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3]
                        ))

                        # Naive sentence splitting: Period, !, ? or essentially newline if it looks like a bullet
                        # If text ends with a punctuation mark, we split.
                        if text.strip().endswith((".", "!", "?")):
                            sentences.append({
                                "text": current_sent_text.strip(),
                                "bboxes": current_sent_bboxes
                            })
                            current_sent_text = ""
                            current_sent_bboxes = []

                # Capture any remaining text in the block as a sentence
                # This handles bullet points that don't end in punctuation but are in separate blocks/lines
                if current_sent_text.strip():
                    sentences.append({
                        "text": current_sent_text.strip(),
                        "bboxes": current_sent_bboxes
                    })

        return sentences

    def chunk(self, pdf_path: str) -> List[Chunk]:
        from sklearn.metrics.pairwise import cosine_similarity

        doc = fitz.open(pdf_path)

        # 1. Extract sentences
        sentences_data = self._extract_sentences(doc)
        if not sentences_data:
            return []

        # 2. Prepare sliding window texts for embedding
        # This adds context to short sentences (like bullets)
        texts_to_embed = []
        for i in range(len(sentences_data)):
            start = max(0, i - self.window_size)
            end = min(len(sentences_data), i + self.window_size + 1)
            window_text = " ".join([s["text"] for s in sentences_data[start:end]])
            texts_to_embed.append(window_text)

        # 3. Compute embeddings
        print(f"Generating embeddings for {len(texts_to_embed)} windows...")
        embeddings = self.model.encode(texts_to_embed)

        # 4. Calculate cosine distances between adjacent sentences
        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            # Clamp sim to [-1, 1] just in case
            sim = max(-1.0, min(1.0, sim))
            dist = 1 - sim
            distances.append(dist)

        # 5. Determine Threshold
        if not distances:
            # Only one sentence
            return [self._create_chunk(sentences_data, 0, 1)]

        # Calculate the percentile threshold
        # e.g., 90th percentile means we only split at the top 10% of distances
        threshold = np.percentile(distances, self.percentile_threshold)
        print(f"Calculated split threshold: {threshold:.4f} (Percentile: {self.percentile_threshold})")

        # 6. Split
        chunks = []
        start_idx = 0

        for i, dist in enumerate(distances):
            if dist > threshold:
                # Split occurring after sentence i (so i is the last sentence of current chunk)
                end_idx = i + 1
                chunks.append(self._create_chunk(sentences_data, start_idx, end_idx))
                start_idx = end_idx

        # Final chunk
        if start_idx < len(sentences_data):
            chunks.append(self._create_chunk(sentences_data, start_idx, len(sentences_data)))

        return chunks

    def _create_chunk(self, sentences_data, start_idx, end_idx):
        subset = sentences_data[start_idx:end_idx]
        combined_text = " ".join([s["text"] for s in subset])
        combined_bboxes = []
        for s in subset:
            combined_bboxes.extend(s["bboxes"])

        return Chunk(
            id=str(uuid.uuid4()),
            text=combined_text,
            bboxes=combined_bboxes,
            metadata={"sentence_count": len(subset)}
        )

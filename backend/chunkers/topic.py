import fitz
import uuid
import numpy as np
from typing import List
from .base import BaseChunker, Chunk, BoundingBox

class TopicChunker(BaseChunker):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", num_topics: int = 5):
        self.model_name = model_name
        self.num_topics = num_topics
        self._model = None

    @property
    def name(self) -> str:
        return f"Topic Chunker (K-Means k={self.num_topics})"

    @property
    def description(self) -> str:
        return "Groups text into topics using K-Means clustering on embeddings. Chunks are non-sequential."

    @property
    def model(self):
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            # Lazy import to speed up app startup
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _extract_sentences(self, doc: fitz.Document) -> List[dict]:
        """Extract sentences/segments with their bounding boxes."""
        # Reuse similar logic to SemanticChunker, extracting sentences as atomic units
        sentences = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue

                # We treat each block as a potential source of sentences
                current_sent_text = ""
                current_sent_bboxes = []

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        bbox = span["bbox"]

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

                        if text.strip().endswith((".", "!", "?")):
                            sentences.append({
                                "text": current_sent_text.strip(),
                                "bboxes": current_sent_bboxes
                            })
                            current_sent_text = ""
                            current_sent_bboxes = []

                if current_sent_text.strip():
                    sentences.append({
                        "text": current_sent_text.strip(),
                        "bboxes": current_sent_bboxes
                    })
        return sentences

    def chunk(self, pdf_path: str) -> List[Chunk]:
        from sklearn.cluster import KMeans

        doc = fitz.open(pdf_path)

        # 1. Extract sentences
        sentences_data = self._extract_sentences(doc)
        if not sentences_data:
            return []

        if len(sentences_data) < self.num_topics:
            # Fallback if fewer sentences than topics
            return self._create_fallback_chunk(sentences_data)

        # 2. Embed sentences
        texts = [s["text"] for s in sentences_data]
        print(f"Generating embeddings for {len(texts)} sentences...")
        embeddings = self.model.encode(texts)

        # 3. Cluster
        print(f"Clustering into {self.num_topics} topics...")
        kmeans = KMeans(n_clusters=self.num_topics, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # 4. Group by Label
        clusters = {i: [] for i in range(self.num_topics)}
        for idx, label in enumerate(labels):
            clusters[label].append(sentences_data[idx])

        # 5. Create Chunks
        chunks = []
        for label in clusters:
            cluster_sentences = clusters[label]
            if not cluster_sentences:
                continue

            # Sort by occurrence in document to keep some reading order within the topic
            # (Our extraction order is sequential, so they are already sorted by page/pos)

            combined_text_parts = []
            combined_bboxes = []

            for s in cluster_sentences:
                combined_text_parts.append(s["text"])
                combined_bboxes.extend(s["bboxes"])

            # Join with a separator to indicate disjointedness
            combined_text = "\n\n---\n\n".join(combined_text_parts)

            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=f"TOPIC {label + 1}:\n" + combined_text,
                bboxes=combined_bboxes,
                metadata={"topic_id": int(label), "sentence_count": len(cluster_sentences)}
            ))

        return chunks

    def _create_fallback_chunk(self, sentences_data):
        combined_text = " ".join([s["text"] for s in sentences_data])
        combined_bboxes = []
        for s in sentences_data:
            combined_bboxes.extend(s["bboxes"])
        return [Chunk(
            id=str(uuid.uuid4()),
            text="Full Document (Too few sentences for topic modeling)",
            bboxes=combined_bboxes,
            metadata={}
        )]

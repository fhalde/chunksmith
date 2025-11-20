import fitz  # pymupdf
from typing import List
import uuid
from .base import BaseChunker, Chunk, BoundingBox

class BasicWordChunker(BaseChunker):
    @property
    def name(self) -> str:
        return "Basic Word Chunker"

    @property
    def description(self) -> str:
        return "Chunks text by words (useful for debugging bounding boxes)."

    def chunk(self, pdf_path: str) -> List[Chunk]:
        doc = fitz.open(pdf_path)
        chunks = []

        # Iterate over pages
        for page_num in range(len(doc)):
            page = doc[page_num]
            # get_text("words") returns list of (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            words = page.get_text("words")

            for w in words:
                x0, y0, x1, y1, text, block, line, word_idx = w

                # Create a chunk for each word
                bbox = BoundingBox(
                    page=page_num,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1
                )

                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    text=text,
                    bboxes=[bbox],
                    metadata={"block": block, "line": line}
                )
                chunks.append(chunk)

        return chunks

class SentenceChunker(BaseChunker):
    @property
    def name(self) -> str:
        return "Sentence Chunker"

    @property
    def description(self) -> str:
        return "Chunks text by sentences (approximate)."

    def chunk(self, pdf_path: str) -> List[Chunk]:
        doc = fitz.open(pdf_path)
        chunks = []

        # Very naive implementation relying on PyMuPDF's text extraction structure
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract dict to get structural info
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue

                current_sentence_text = ""
                current_sentence_bboxes = []

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        bbox = span["bbox"]

                        current_sentence_text += text + " "
                        current_sentence_bboxes.append(BoundingBox(
                            page=page_num,
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3]
                        ))

                        # Simple splitting on period (very naive)
                        if text.strip().endswith("."):
                            chunk = Chunk(
                                id=str(uuid.uuid4()),
                                text=current_sentence_text.strip(),
                                bboxes=current_sentence_bboxes,
                                metadata={}
                            )
                            chunks.append(chunk)
                            current_sentence_text = ""
                            current_sentence_bboxes = []

                # Add remaining text in block
                if current_sentence_text.strip():
                    chunk = Chunk(
                        id=str(uuid.uuid4()),
                        text=current_sentence_text.strip(),
                        bboxes=current_sentence_bboxes,
                        metadata={}
                    )
                    chunks.append(chunk)

        return chunks


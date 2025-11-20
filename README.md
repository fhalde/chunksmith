# ChunkSmith

**ChunkSmith** is a specialized workbench for Chunk Engineers. It allows you to visualize, test, and refine PDF chunking algorithms.

Designed for developers building RAG (Retrieval-Augmented Generation) pipelines, ChunkSmith provides a visual interface to see exactly *where* and *how* your documents are being split.

## Installation

ChunkSmith is built with Python and uses `uv` for fast dependency management.

### Prerequisites
- Python 3.12+
- `uv` (Universal Package Manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/fhalde/chunksmith.git
   cd chunksmith
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Run the application**
   ```bash
   uv run main.py
   ```

## Included Algorithms

ChunkSmith comes with several reference implementations to get you started:

1. **Basic Word Chunker**:
   - Treats every individual word as a chunk.
   - *Use case*: Debugging bounding box accuracy and coordinate systems.

2. **Sentence Chunker**:
   - Splits text by sentence boundaries using PyMuPDF.
   - *Use case*: Standard NLP tasks where sentence-level granularity is needed.

3. **Semantic Chunker (Percentile)**:
   - Uses `sentence-transformers` to generate embeddings for sliding windows of text.
   - Calculates cosine distance between adjacent sentences.
   - Dynamically splits at the **90th percentile** of distances (the "peaks" of semantic change).
   - *Use case*: Resumes, scientific papers, or structured documents where you want to capture distinct sections (e.g., "Experience" vs "Education").

4. **Topic Chunker (K-Means)**:
   - Clusters sentences based on semantic similarity using K-Means.
   - **Non-sequential**: Can group a paragraph from Page 1 and a paragraph from Page 10 into the same chunk if they discuss the same topic.
   - *Use case*: Topic modeling, extracting specific themes (e.g., "Legal Disclaimers" scattered throughout a contract).

## For Chunk Engineers: Adding a New Algorithm

1. Create a new file in `backend/chunkers/` (e.g., `my_chunker.py`).
2. Create a class that inherits from `BaseChunker`.
3. Implement the `chunk` method.

```python
from typing import List
from .base import BaseChunker, Chunk, BoundingBox

class MyCustomChunker(BaseChunker):
    @property
    def name(self) -> str:
        return "My Custom Logic"

    @property
    def description(self) -> str:
        return "Splits by... magic?"

    def chunk(self, pdf_path: str) -> List[Chunk]:
        # Your logic here using pymupdf (fitz)
        return []
```

4. Register your new chunker in `backend/api.py`.

```python
from .chunkers.my_chunker import MyCustomChunker

# ... inside Api.__init__
self.chunkers = {
    # ...
    "My Custom Logic": MyCustomChunker(),
}
```

5. Restart the app. Your new algorithm will appear in the dropdown.

## License

MIT


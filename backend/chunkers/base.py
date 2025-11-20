from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class BoundingBox:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float

@dataclass
class Chunk:
    id: str
    text: str
    bboxes: List[BoundingBox]  # A chunk might span multiple lines/areas
    metadata: Dict[str, Any] = None

class BaseChunker(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the display name of the algorithm."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the algorithm."""
        pass

    @abstractmethod
    def chunk(self, pdf_path: str) -> List[Chunk]:
        """Process the PDF and return a list of chunks with bounding boxes."""
        pass


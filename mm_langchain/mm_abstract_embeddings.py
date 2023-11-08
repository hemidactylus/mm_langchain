from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from .mm_types import MMContent, MMStoredDocument


class MMEmbeddings(ABC):
    """Abstract multimodal embeddings."""

    modality_type_map: Dict[str, type]

    @property
    def modalities(self) -> Set[str]:
        return set(self.modality_type_map.keys())

    @abstractmethod
    def embed_by_modality(self, modality: str, value: Any) -> List[float]:
        """
        Single-modality embedding computation.
            Note: perhaps not used by all embeddings (e.g. 'holistic' models)
            or maybe not worth having it here at all ...
        """

    def embed_one(self, content: MMContent) -> List[float]:
        """
        Embed a single piece of multimodal content.
        For now, the merging policy is hardcoded here.
        """
        assert(content != {})
        assert len(content.keys() - self.modality_type_map.keys()) == 0
        for modality, value in content.items():
            assert isinstance(value, self.modality_type_map[modality])
        #
        vectors_map = {
            modality: self.embed_by_modality(modality, value)
            for modality, value in content.items()
        }
        # average:
        num_modalities = len(vectors_map)
        return [sum(xs)/num_modalities for xs in zip(*vectors_map.values())]

    def embed_many(self, contents: List[MMContent]) -> List[List[float]]:
        """Embed a list of contents."""
        return [self.embed_one(content) for content in contents]


# this concerns the layer between the mm vector store and the reader-writer
class MMContentSerializer(ABC):

    @property
    def modalities(self) -> Set[str]:
        raise NotImplementedError

    @abstractmethod
    def serialize(self, content: MMContent) -> Dict[str, str]:
        """make all parts of the content into a string."""

    def unserialize_stored(self, stored: Dict[str, str], metadata: Optional[dict]) -> MMStoredDocument:
        """default is to leave as is."""
        return cast(MMStoredDocument, stored)

from typing import Any, Dict, List, Optional, Set

from PIL.Image import Image as PILImageType

from langchain.pydantic_v1 import BaseModel, Field

from .mm_types import MMContent, MMStoredDocument
from .mm_abstract_embeddings import MMEmbeddings, MMContentSerializer


class MMHuggingFaceEmbeddings(BaseModel, MMEmbeddings):
    """
    Modality type map:
        "text" => str
        "image" => PIL.Image.Image
    """

    client: Any  #: :meta private:
    model_name: str
    cache_folder: Optional[str] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    modality_type_map = {
        "text": str,
        "image": PILImageType,
    }

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        try:
            import sentence_transformers

        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        self.client = sentence_transformers.SentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

    def embed_by_modality(self, modality: str, value: Any) -> List[float]:
        if modality == "text":
            return self.client.encode([value])[0]
        elif modality == "image":
            return self.client.encode([value])[0]
        else:
            raise ValueError(f"Unknown modality '{modality}'")

class MMImageTextSerializer(MMContentSerializer):

    @property
    def modalities(self) -> Set[str]:
        return {"text", "image"}

    def serialize(self, content: MMContent) -> Dict[str, str]:
        # dirty implementation
        return {
            modality: value if modality == "text" else "(an image)"
            for modality, value in content.items()
        }

    def unserialize_stored(self, stored: Dict[str, str], metadata: Optional[dict]) -> MMStoredDocument:
        # dirty implementation
        return MMStoredDocument(
            content={
                modality: value if modality == "text" else metadata.get("image_path", "<IMAGE>")
                for modality, value in stored.items()
            },
            metadata=metadata,
        )

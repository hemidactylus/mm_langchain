from typing import Any, Dict, Literal, Optional, Set, Tuple

from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Field

# stored "thing" on a vector store
# (id, stringy blob, metadata, similarity score)
DefaultVSearchResult = Tuple[str, str, Optional[Dict], float]

# MULTIMODAL-SPECIFIC CONSTRUCTS:

# multimodal "content"
MMContent = Dict[str, Any]

# MMDocument is a container for multimodal stuff to be exchanged between
# mm vector stores and mm embeddings
class MMDocument(Serializable):
    content: MMContent
    metadata: dict = Field(default_factory=dict)

    type: Literal["MMDocument"] = "MMDocument"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False  # for now...

# as a matter of fact, it's an alias for now
MMStoredDocument = MMDocument

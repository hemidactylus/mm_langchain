from typing import Any, Dict, Optional, Tuple


# multimodal "content"
MMContent = Dict[str, Any]


# stored "thing" on a vector store
# (id, stringy blob, metadata, similarity score)
DefaultVSearchResult = Tuple[str, str, Optional[Dict], float]

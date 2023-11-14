from abc import ABC, abstractmethod
from typing import List, Optional

from ..mm_types import MMDocument

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class MMBaseLoader(ABC):

    @abstractmethod
    def load(self) -> List[MMDocument]:
        """We don't bother with lazy loading for now"""

    def load_and_split(
        self,
        text_splitter: Optional[TextSplitter] = None,
    ) -> List[MMDocument]:
        if text_splitter is None:
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter(
                # small, for demonstration purposes
                chunk_size=256,
                chunk_overlap=64,
            )
        else:
            _text_splitter = text_splitter
        # TO BE FIXED in multimodal logic...
        docs = self.load()
        return _text_splitter.split_documents(docs)

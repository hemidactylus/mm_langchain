from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, List, Optional, TypeVar

## Components from langchain
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings

# i.e. either str or MMContent in the two cases at hand
# C = TypeVar('C')

# i.e. whatever comes out of the vector store backend:
#   (e.g. a DefaultVSearchResult)
S = TypeVar('S')


class VectorReaderWriter(ABC, Generic[S]):

    @abstractmethod
    def store_contents(self, contents: Iterable[str], vectors: Iterable[List[float]], metadatas: Optional[Iterable[dict]] = None, **kwargs: Any) -> List[str]:
        """Actual storing to backend. the "contents" are stringy blobs, no questions asked."""

    @abstractmethod
    def search_by_vector(self, vector: List[float], k: int = 4, **kwargs: Any) -> List[S]:
        """run an ANN search and return an S for each returned entry"""


class VectorStore(Generic[S]):

    vector_reader_writer: VectorReaderWriter[S]
    embedding: Embeddings

    @property
    def embeddings(self) -> Optional[Embeddings]:
        raise NotImplementedError


    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Return docs most similar to query.
        The implementation depends at least on what `S` is)
        """

    def add_texts(self, texts: List[str], metadatas: Optional[List[Optional[dict]]] = None, **kwargs: Any) -> List[str]:
        """run texts through the embedding and store the full resulting entries."""
        embedding_vectors = self.embedding.embed_documents(texts)
        metadatas0 = [md or {} for md in metadatas]
        return self.vector_reader_writer.store_contents(
            contents=texts,
            vectors=embedding_vectors,
            metadatas=metadatas0,
            **kwargs,
        )

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

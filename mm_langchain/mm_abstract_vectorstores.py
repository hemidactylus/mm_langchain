import json
from abc import ABC, abstractmethod
from typing import Any, cast, Generic, Iterable, List, Optional, Set, TypeVar

## Components from langchain
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings

from .mm_abstract_embeddings import MMEmbeddings, MMContentSerializer
from .mm_types import MMContent, MMDocument, MMStoredDocument

# i.e. either str or MMContent in the two cases at hand
# C = TypeVar('C')

# i.e. whatever comes out of the vector store backend:
#   (e.g. a DefaultVSearchResult)
S = TypeVar('S')


class VectorReaderWriter(ABC, Generic[S]):

    @abstractmethod
    def store_contents(self, contents_str: Iterable[str], vectors: Iterable[List[float]], metadatas: Optional[Iterable[dict]] = None, **kwargs: Any) -> List[str]:
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
        self, query: MMContent, k: int = 4, **kwargs: Any
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
            contents_str=texts,
            vectors=embedding_vectors,
            metadatas=metadatas0,
            **kwargs,
        )

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)


class MMVectorStore(ABC, Generic[S]):

    vector_reader_writer: VectorReaderWriter[S]
    embedding: MMEmbeddings
    content_serializer: MMContentSerializer

    def __init__(
        self,
        *pargs: Any,
        vector_reader_writer: VectorReaderWriter[S],
        embedding: MMEmbeddings,
        content_serializer: MMContentSerializer,
        **kwargs: Any,
    ) -> None:
        self.vector_reader_writer = vector_reader_writer
        self.embedding = embedding
        self.content_serializer = content_serializer
        assert len(self.embedding.modalities - self.content_serializer.modalities) == 0

    @property
    def embeddings(self) -> Optional[MMEmbeddings]:
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self, query: MMContent, k: int = 4, **kwargs: Any
    ) -> List[MMStoredDocument]:
        """
        Return docs most similar to query.
        The implementation depends at least on what `S` is)
        """

    def add_contents(self, contents: List[MMContent], metadatas: Optional[List[Optional[dict]]] = None, **kwargs: Any) -> List[str]:
        """run contexts through the embedding and store the full resulting entries."""
        embedding_vectors = self.embedding.embed_many(contents)
        metadatas0 = [md or {} for md in metadatas]
        contents_str = [
            json.dumps(serialized)
            for serialized in (
                self.content_serializer.serialize(content)
                for content in contents
            )
        ]
        return self.vector_reader_writer.store_contents(
            contents_str=contents_str,
            vectors=embedding_vectors,
            metadatas=metadatas0,
            **kwargs,
        )

    def add_documents(self, documents: List[MMDocument], **kwargs: Any) -> List[str]:
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        return self.add_contents(contents, metadatas, **kwargs)

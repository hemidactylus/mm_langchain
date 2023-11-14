import uuid
from typing import Any, Dict, Iterable, List, Optional

## Components from langchain
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings

from .mm_types import DefaultVSearchResult, MMContent, MMStoredDocument
from .mm_abstract_vectorstores import (
    MMVectorStore,
    VectorReaderWriter,
    VectorStore,
)
from .mm_abstract_embeddings import MMEmbeddings, MMContentSerializer

from cassio.table import MetadataVectorCassandraTable


class CassandraVectorReaderWriter(VectorReaderWriter[DefaultVSearchResult]):
    def __init__(
        self,
        table_name: str,
        vector_dimension: int,
    ) -> None:
        self.table = MetadataVectorCassandraTable(
            table=table_name,
            vector_dimension=vector_dimension,
        )

    def store_contents(
        self,
        contents_str: Iterable[str],
        vectors: Iterable[List[float]],
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # silly reference implementation
        contents0 = list(contents_str)
        vectors0 = list(vectors)
        metadatas0 = list(metadatas) if metadatas else [{}] * len(contents0)
        ids0 = list(ids) if ids else [uuid.uuid4().hex for _ in contents0]
        inserteds = []
        for xco, xve, xme, xid in zip(contents0, vectors0, metadatas0, ids0):
            self.table.put(
                row_id=xid,
                body_blob=xco,
                vector=xve,
                metadata=xme,
            )
            inserteds.append(xid)
        return inserteds

    def search_by_vector(
        self,
        vector: List[float],
        k: int = 4,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[DefaultVSearchResult]:
        return [
            (
                result["row_id"],
                result["body_blob"],
                result["metadata"],
                result["distance"],
            )
            for result in self.table.metric_ann_search(
                vector=vector,
                n=k,
                metadata=metadata,
                metric="cos",
                metric_threshold=None,
            )
        ]

    def clear(self):
        self.table.clear()


class Cassandra(VectorStore[DefaultVSearchResult]):
    @staticmethod
    def _filter_to_metadata(filter_dict: Optional[Dict[str, str]]) -> Dict[str, Any]:
        if filter_dict is None:
            return {}
        else:
            return filter_dict

    def __init__(
        self,
        embedding: Embeddings,
        table_name: str,
    ):
        self.embedding = embedding
        self._embedding_dimension = len(
            embedding.embed_query("This is a sample sentence.")
        )
        self.vector_reader_writer = CassandraVectorReaderWriter(
            table_name=table_name,
            vector_dimension=self._embedding_dimension,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        search_metadata = self._filter_to_metadata(filter)
        search_vector = self.embedding.embed_query(query)
        return [
            Document(page_content=rbl, metadata=rme)
            for (rid, rbl, rme, rsi) in self.vector_reader_writer.search_by_vector(
                vector=search_vector,
                k=k,
                metadata=search_metadata,
                **kwargs,
            )
        ]


class MMCassandra(MMVectorStore[DefaultVSearchResult]):
    @staticmethod
    def _filter_to_metadata(filter_dict: Optional[Dict[str, str]]) -> Dict[str, Any]:
        if filter_dict is None:
            return {}
        else:
            return filter_dict

    def __init__(
        self,
        embedding: MMEmbeddings,
        content_serializer: MMContentSerializer,
        table_name: str,
        *pargs,
        **kwargs,
    ) -> None:
        self._embedding_dimension = len(
            embedding.embed_one({"text": "This is a sample sentence."})
        )
        vector_rw = CassandraVectorReaderWriter(
            table_name=table_name,
            vector_dimension=self._embedding_dimension,
        )
        super().__init__(
            vector_reader_writer=vector_rw,
            embedding=embedding,
            content_serializer=content_serializer,
        )

    @property
    def embeddings(self) -> Optional[MMEmbeddings]:
        raise NotImplementedError

    def similarity_search(
        self,
        query: MMContent,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[MMStoredDocument]:
        """Return (mm) docs most similar to query."""
        search_metadata = self._filter_to_metadata(filter)
        search_vector = self.embedding.embed_one(query)
        return [
            MMStoredDocument(
                content=self.content_serializer.deserialize_stored_str_to_content(
                    rbl, metadata=rme
                ),
                metadata=rme,
            )
            for (rid, rbl, rme, rsi) in self.vector_reader_writer.search_by_vector(
                vector=search_vector,
                k=k,
                metadata=search_metadata,
                **kwargs,
            )
        ]

    def clear(self):
        self.vector_reader_writer.clear()

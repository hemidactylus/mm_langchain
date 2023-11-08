from typing import Any, Dict, Iterable, List, Optional, Union

## Components from langchain
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings

from .abstract_vectorstores import VectorStore, VectorReaderWriter
from .mm_types import DefaultVSearchResult

from cassio.table import MetadataVectorCassandraTable


class CassandraVectorReaderWriter(VectorReaderWriter[DefaultVSearchResult]):

    def __init__(
        self,
        table_name: str,
        vector_dimension: str,
    ) -> None:
        self.table = MetadataVectorCassandraTable(
            table=table_name,
            vector_dimension=vector_dimension,
        )

    def store_contents(self, contents: Iterable[str], vectors: Iterable[List[float]], metadatas: Optional[Iterable[dict]] = None, ids: Optional[Iterable[str]] = None, **kwargs: Any) -> List[str]:
        # silly reference implementation
        contents0 = list(contents)
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

    def search_by_vector(self, vector: List[float], k: int = 4, metadata: Optional[dict] = None, **kwargs: Any) -> List[DefaultVSearchResult]:
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


class Cassandra(VectorStore[DefaultVSearchResult]):

    _embedding_dimension: Union[int, None]

    @staticmethod
    def _filter_to_metadata(filter_dict: Optional[Dict[str, str]]) -> Dict[str, Any]:
        if filter_dict is None:
            return {}
        else:
            return filter_dict

    def _get_embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            self._embedding_dimension = len(
                self.embedding.embed_query("This is a sample sentence.")
            )
        return self._embedding_dimension
    
    def __init__(
        self,
        embedding: Embeddings,
        table_name: str,
    ):
        self.embedding = embedding
        self._embedding_dimension = None
        self.vector_reader_writer = CassandraVectorReaderWriter(
            table_name=table_name,
            vector_dimension=self._get_embedding_dimension(),
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

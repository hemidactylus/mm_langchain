import json
from typing import Any, Iterable, List, Optional, Type, TypeVar

from langchain.schema.embeddings import Embeddings as BaseEmbeddings
from langchain.schema.vectorstore import VectorStore as BaseVectorStore

from ..mm_types import MMContent, MMStoredDocument
from ..mm_abstract_embeddings import MMEmbeddings, MMContentSerializer
from ..mm_abstract_vectorstores import MMVectorStore, VectorReaderWriter
from .utils import compress_vector, deflate_vector


VST = TypeVar("VST", bound="BaseVectorStore")


class DummyVectorReaderWriter(VectorReaderWriter):
    def store_contents(
        self,
        contents_str: Iterable[str],
        vectors: Iterable[List[float]],
        metadatas: Optional[Iterable[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError("Placeholder")

    def search_by_vector(self, vector: List[float], k: int = 4, **kwargs: Any) -> List:
        raise NotImplementedError("Placeholder")


def _wrap_for_base_vectorstore(
    content: MMContent, emb_vector: List[float], content_serializer: MMContentSerializer
) -> str:
    vector_dimension = len(emb_vector)
    return json.dumps(
        {
            "stored": content_serializer.serialize_content(content),
            "embedding_vector": compress_vector(emb_vector, vector_dimension),
            "vector_dimension": vector_dimension,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _unwrap_from_base_vectorstore(
    wrapped_content_str: str,
    metadata: Optional[dict],
    content_serializer: MMContentSerializer,
) -> MMContent:
    stored = json.loads(wrapped_content_str)["stored"]
    return content_serializer.deserialize_stored(
        stored=stored,
        metadata=metadata,
    )


class DTMMVectorStore(MMVectorStore):
    base_vector_store: BaseVectorStore

    def __init__(
        self,
        embedding: MMEmbeddings,
        content_serializer: MMContentSerializer,
        base_vector_store: BaseVectorStore,
    ):
        self.base_vector_store = base_vector_store
        # this is going to be bypassed throughout:
        vector_rw = DummyVectorReaderWriter()
        super().__init__(
            vector_reader_writer=vector_rw,
            embedding=embedding,
            content_serializer=content_serializer,
        )

    def add_contents(
        self,
        contents: List[MMContent],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        embedding_vectors = self.embedding.embed_many(contents)
        wrapped_contents_str = [
            _wrap_for_base_vectorstore(
                content=content,
                emb_vector=emb_vector,
                content_serializer=self.content_serializer,
            )
            for content, emb_vector in zip(contents, embedding_vectors)
        ]
        return self.base_vector_store.add_texts(
            texts=wrapped_contents_str,
            metadatas=metadatas,
            **kwargs,
        )

    def similarity_search(
        self, query: MMContent, k: int = 4, **kwargs: Any
    ) -> List[MMStoredDocument]:
        search_vector = self.embedding.embed_one(query)
        wrapped_search_content_str = _wrap_for_base_vectorstore(
            content=query,
            emb_vector=search_vector,
            content_serializer=self.content_serializer,
        )
        base_store_results = self.base_vector_store.similarity_search_with_score(
            query=wrapped_search_content_str,
            k=k,
            **kwargs,
        )
        results: List[MMStoredDocument] = []
        for base_document, score in base_store_results:
            wrapped_content_str = base_document.page_content
            metadata = base_document.metadata
            #
            mmdoc = MMStoredDocument(
                content=_unwrap_from_base_vectorstore(
                    wrapped_content_str=wrapped_content_str,
                    metadata=metadata,
                    content_serializer=self.content_serializer,
                ),
                metadata=metadata,
            )
            results.append(mmdoc)
            #
        return results


class DTPassthroughEmbeddings(BaseEmbeddings):
    def __init__(self, embedding_dimension: int) -> None:
        self.embedding_dimension = embedding_dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """these texts are `wrapped_contents_str` kind of texts"""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        We try to json loads the input. If this fails,
        assume it's a dummy call to probe the dimension.
        (can't think of anything better here).
        """
        try:
            contents_obj = json.loads(text)
            vector_dimension = contents_obj["vector_dimension"]
            return deflate_vector(contents_obj["embedding_vector"], vector_dimension)
        except Exception:
            return [0.0] * self.embedding_dimension


def duct_tape_make_multimodal(
    *pargs: Any,
    base_vectorstore_class: Type[VST],
    embedding: MMEmbeddings,
    content_serializer: MMContentSerializer,
    test_mm_content: MMContent = {"text": "This is a sample sentence."},
    **kwargs: Any,
) -> DTMMVectorStore:
    """
    create a multimodal-ready vector store by duct-taping it onto
    a regular vector store. Not the cleanest way, but it enables
    multi-modal for arbitrary stores right now.

    Requirements:
        base_vectorstore_class's constructor must have `embedding` param
        its `similarity_search_with_score` must have `query` and `k` params
    """

    # no way to set the dimension explicitly in the base v.store...
    embedding_dimension = len(embedding.embed_one(test_mm_content))
    passthrough_embedding = DTPassthroughEmbeddings(
        embedding_dimension=embedding_dimension
    )
    # THIS DOES NOT TYPECHECK, and we know it
    base_vector_store = base_vectorstore_class(
        embedding=passthrough_embedding, **kwargs
    )  # type: ignore

    return DTMMVectorStore(
        embedding=embedding,
        content_serializer=content_serializer,
        base_vector_store=base_vector_store,
    )

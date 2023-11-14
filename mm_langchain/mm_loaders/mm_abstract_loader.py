from abc import ABC, abstractmethod
from typing import List, Optional

from ..mm_types import MMDocument

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.docstore.document import Document


class MMDisjointBaseLoader(ABC):
    """
    'disjoint' as in: produces a list of MMDocuments that are
    each *either* a text or an image. Demo purposes.
    """

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
        docs0 = self.load()
        # Draft implementation - temporarily rewrapping as 'Document's
        documents = []
        doc_buffer = []
        for doc0 in docs0:
            if "text" in doc0.content:
                if "image" in doc0.content:
                    raise ValueError("Multi-modal doc encountered.")
                doc_buffer.append(doc0)
            elif "image" in doc0.content:
                # flush and split the buffer (with the rewrapping trick)
                if doc_buffer:
                    t_docs = [
                        Document(
                            page_content=b_doc.content["text"], metadata=b_doc.metadata
                        )
                        for b_doc in doc_buffer
                    ]
                    split_t_docs = _text_splitter.split_documents(t_docs)
                    documents += [
                        MMDocument(
                            content={"text": split_t_doc.page_content},
                            metadata=split_t_doc.metadata,
                        )
                        for split_t_doc in split_t_docs
                    ]
                    doc_buffer = []
                # add this image
                documents.append(doc0)
        # flush the remaining part
        if doc_buffer:
            t_docs = [
                Document(page_content=b_doc.content["text"], metadata=b_doc.metadata)
                for b_doc in doc_buffer
            ]
            split_t_docs = _text_splitter.split_documents(t_docs)
            documents += [
                MMDocument(
                    content={"text": split_t_doc.page_content},
                    metadata=split_t_doc.metadata,
                )
                for split_t_doc in split_t_docs
            ]
            doc_buffer = []

        return documents

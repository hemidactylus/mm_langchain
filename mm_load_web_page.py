import os

import cassio

from mm_langchain.mm_vectorstores import MMCassandra
from mm_langchain.mm_huggingface_embeddings import (
    MMHuggingFaceEmbeddings,
    MMImageTextSerializer,
)
from mm_langchain.mm_loaders.mm_web_page_loader import MMDisjointWebBaseLoader


cassio.init(
    database_id=os.environ["ASTRA_DB_ID"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
)

vector_store_name = "mm_test"
web_url = "https://gist.github.com/hemidactylus/f2371591318128e3af9840d86699a9e9"

mm_embeddings = MMHuggingFaceEmbeddings(model_name="clip-ViT-B-32")
mm_vectorstore = MMCassandra(
    embedding=mm_embeddings,
    content_serializer=MMImageTextSerializer(),
    table_name=vector_store_name,
)
mm_vectorstore.clear()

loader = MMDisjointWebBaseLoader(web_url)
added = mm_vectorstore.add_documents(loader.load_and_split())

print(f"Added {len(added)} documents from {web_url}.")

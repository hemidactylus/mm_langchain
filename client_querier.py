from pathlib import Path
import os

from multimodal_support import (
    MultiModalHuggingFaceEmbeddings,
    MultiModalCassandra,
)
import cassio


cassio.init(
    database_id=os.environ["ASTRA_DB_ID"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
)

vector_store_name = 'mm_test'

clip_embeddings = MultiModalHuggingFaceEmbeddings(model_name="clip-ViT-B-32")
vectorstore = MultiModalCassandra(embedding=clip_embeddings, table_name=vector_store_name, session=None, keyspace=None)

query_string = "a house in modern style"

results = vectorstore.similarity_search(query_string, k=1)
if results == []:
    print("\n\n** It appears that the vector store is empty. Please populate it first **\n")
else:
    result = results[0]
    print(result.page_content, result.metadata)

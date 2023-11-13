import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document

from mm_langchain.mm_vectorstores import Cassandra


import cassio


cassio.init(
    database_id=os.environ["ASTRA_DB_ID"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
)

vector_store_name = 'va_test'

v_embeddings = OpenAIEmbeddings()
v_vectorstore = Cassandra(embedding=v_embeddings, table_name=vector_store_name)


for doc in v_vectorstore.similarity_search("Who can be my ally here?", k=1):
    print(doc)

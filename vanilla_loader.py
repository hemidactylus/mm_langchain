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

vector_store_name = "va_test"

v_embeddings = OpenAIEmbeddings()
v_vectorstore = Cassandra(embedding=v_embeddings, table_name=vector_store_name)


ids1 = v_vectorstore.add_documents(
    documents=[
        Document(page_content="Beware of the spiderless house."),
        Document(
            page_content="Argiope is a friendly spider in the meadows.",
            metadata={"family": "araneidae"},
        ),
    ],
    ids=["bew", "arg"],
)
print("ids1 = ", ids1)
ids2 = v_vectorstore.add_texts(
    texts=[
        "Spiders have many eyes but often poor sight.",
        "Salticidae can see very well.",
    ],
    metadatas=[
        None,
        {"family": "salticidae"},
    ],
    ids=["eye", "sal"],
)
print("ids2 = ", ids2)

import os

from PIL import Image

import cassio

from mm_langchain.mm_vectorstores import MMCassandra
from mm_langchain.mm_huggingface_embeddings import MMHuggingFaceEmbeddings, MMImageTextSerializer
from mm_langchain.mm_types import MMContent


cassio.init(
    database_id=os.environ["ASTRA_DB_ID"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
)

vector_store_name = 'mm_test'

mm_embeddings = MMHuggingFaceEmbeddings(model_name="clip-ViT-B-32")
mm_vectorstore = MMCassandra(
    embedding=mm_embeddings,
    content_serializer=MMImageTextSerializer(),
    table_name=vector_store_name,
)

# querying in a couple of ways

print("\nText query on image entries.")
query1: MMContent = {"text": "An iguana on a bike"}
results1 = mm_vectorstore.similarity_search(query1, k=1)
print("Results:")
for result1 in results1:
    print(f" * Content={result1.content} [metadata={result1.metadata}]")

query_image = "images/query/shining_sun.jpg"
print("\nImage query on image entries.")
query2: MMContent = {"image": Image.open(query_image)}
results2 = mm_vectorstore.similarity_search(query2, k=1)
print("Results:")
for result2 in results2:
    print(f" * Content={result2.content} [metadata={result2.metadata}]")

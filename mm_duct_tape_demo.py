import os

from PIL import Image

from langchain.vectorstores import Cassandra as OriginalCassandra

from mm_langchain.mm_huggingface_embeddings import (
    MMHuggingFaceEmbeddings,
    MMImageTextSerializer,
)
from mm_langchain.mm_types import MMDocument, MMContent
from mm_langchain.mm_duct_tape import duct_tape_make_multimodal

import cassio


filenames = [
    "images/iguana_on_a_bike.jpg",
    "images/mad_scientist.jpg",
    "images/modern_house.jpg",
    "images/sunset.jpg",
]

cassio.init(
    database_id=os.environ["ASTRA_DB_ID"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
)

dt_vector_store_name = "dt_mm_test"

mm_embeddings = MMHuggingFaceEmbeddings(model_name="clip-ViT-B-32")

dt_mm_vectorstore = duct_tape_make_multimodal(
    base_vectorstore_class=OriginalCassandra,
    embedding=mm_embeddings,
    content_serializer=MMImageTextSerializer(),
    table_name=dt_vector_store_name,
    session=None,
    keyspace=None,
)

dt_mm_vectorstore.base_vector_store.clear()

print("\nINSERTING\n")

# Insertions through store_contents: the first two images
insertion1 = dt_mm_vectorstore.add_contents(
    contents=[
        {"image": Image.open(filenames[0])},
        {"image": Image.open(filenames[1])},
    ],
    metadatas=[
        {"image_path": filenames[0], "category": "animals"},
        {"category": "people"},
    ],
    ids=["iguana", "scientist_012"],
)
print(f"insertion1 = {insertion1}")

# Insertion through add_documents:
insertion2 = dt_mm_vectorstore.add_documents(
    documents=[
        MMDocument(content={"image": Image.open(filenames[2])}),
        MMDocument(
            content={"image": Image.open(filenames[3])},
            metadata={"image_path": filenames[3]},
        ),
    ]
)
print(f"insertion2 = {insertion2}")

print("\nQUERYING\n")

print("\nText query on image entries.")
query1: MMContent = {"text": "An iguana on a bike"}
results1 = dt_mm_vectorstore.similarity_search(query1, k=1)
print("Results:")
for result1 in results1:
    print(f" * Content={result1.content} [metadata={result1.metadata}]")

print("\nText query on image entries (result lacks rich metadata).")
query2: MMContent = {"text": "My oh-so-modern hous"}
results2 = dt_mm_vectorstore.similarity_search(query2, k=1)
print("Results:")
for result2 in results2:
    print(f" * Content={result2.content} [metadata={result2.metadata}]")

query_image = "images/query/shining_sun.jpg"
print("\nImage query on image entries.")
query3: MMContent = {"image": Image.open(query_image)}
results3 = dt_mm_vectorstore.similarity_search(query3, k=1)
print("Results:")
for result3 in results3:
    print(f" * Content={result3.content} [metadata={result3.metadata}]")

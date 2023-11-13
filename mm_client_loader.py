import os

from PIL import Image

import cassio

from mm_langchain.mm_vectorstores import MMCassandra
from mm_langchain.mm_huggingface_embeddings import (
    MMHuggingFaceEmbeddings,
    MMImageTextSerializer,
)
from mm_langchain.mm_types import MMDocument


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

vector_store_name = "mm_test"

mm_embeddings = MMHuggingFaceEmbeddings(model_name="clip-ViT-B-32")
mm_vectorstore = MMCassandra(
    embedding=mm_embeddings,
    content_serializer=MMImageTextSerializer(),
    table_name=vector_store_name,
)
mm_vectorstore.clear()

# Insertions through store_contents: the first two images
insertion1 = mm_vectorstore.add_contents(
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
insertion2 = mm_vectorstore.add_documents(
    documents=[
        MMDocument(content={"image": Image.open(filenames[2])}),
        MMDocument(
            content={"image": Image.open(filenames[3])},
            metadata={"image_path": filenames[3]},
        ),
    ]
)
print(f"insertion2 = {insertion2}")

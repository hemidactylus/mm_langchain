from mm_langchain.mm_loaders.mm_web_page_loader import MMWebBaseLoader

loader = MMWebBaseLoader("https://gist.github.com/hemidactylus/f2371591318128e3af9840d86699a9e9")

docs = loader.load()

for doc_i, doc in enumerate(docs):
    print(f"  [doc {doc_i}] {', '.join(sorted(doc.content.keys()))}")
    if "text" in doc.content:
        print(f"        text => {doc.content['text'][:48]} ...")
    if "image" in doc.content:
        print(f"        image => {doc.content['image'].size} (from {doc.metadata['image_url']})")

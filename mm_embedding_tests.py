from PIL import Image

from mm_langchain.mm_huggingface_embeddings import MMHuggingFaceEmbeddings


filenames = [
    "images/iguana_on_a_bike.jpg",
    "images/mad_scientist.jpg",
    "images/modern_house.jpg",
    "images/sunset.jpg",
]

mm_embeddings = MMHuggingFaceEmbeddings(model_name="clip-ViT-B-32")

image_contents = [{"image": Image.open(fn)} for fn in filenames]


def cos(v1, v2):
    return (
        sum(x * y for x, y in zip(v1, v2))
        / (sum(x * x for x in v1) * sum(y * y for y in v2)) ** 0.5
    )


def cos_sim(t1=None, ip1=None, t2=None, ip2=None):
    ct1 = {
        k: v
        for k, v in {
            "text": t1,
            "image": Image.open(ip1) if ip1 else None,
        }.items()
        if v
    }
    ct2 = {
        k: v
        for k, v in {
            "text": t2,
            "image": Image.open(ip2) if ip2 else None,
        }.items()
        if v
    }
    vecs = mm_embeddings.embed_many([ct1, ct2])
    return cos(vecs[0], vecs[1])


"""
Notes:
    Not magically well normalized when using different choices of modalities

    cos_sim("an iguana on a bike", None, "A cartoon of a mad scientist", None)
    cos_sim("an iguana on a bike", None, "A house in modern avant-garde architectural style", None)
    cos_sim("an iguana on a bike", None, "A sunset on the sea, with vivid colors, probably photoshopped", None)
    # 0.53, 0.52, 0.49

    cos_sim(None, filenames[0], None, filenames[1])
    cos_sim(None, filenames[0], None, filenames[2])
    cos_sim(None, filenames[0], None, filenames[3])
    # 0.49, 0.58, 0.36

    # mixed
    cos_sim("an iguana on a bike", None, None, filenames[0])
    # 0.36 (!!!)

    cos_sim("an iguana on a bike", None, None, filenames[1])
    cos_sim("an iguana on a bike", None, None, filenames[2])
    cos_sim("an iguana on a bike", None, None, filenames[3])
    # 0.13, 0.10, 0.16

    # More-than-one:
    cos_sim("an iguana on a bike", None, "A tiny animal riding a miniature bicycle", filenames[0])
    # 0.63

    It is beyond our scope NOW to explore normalizations and pre-post processing of vectors etc
"""

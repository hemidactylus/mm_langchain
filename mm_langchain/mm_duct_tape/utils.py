from typing import List
import struct
import base64


def compress_vector(vector: List[float], n: int) -> str:
    return base64.b64encode(struct.pack("%if" % n, *vector)).decode()


def deflate_vector(compressed: str, n: int) -> List[float]:
    return list(struct.unpack("%if" % n, base64.b64decode(compressed)))

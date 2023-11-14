from typing import Any, Iterable, List
import requests
import io

from PIL import Image
from PIL.Image import Image as PILImageType
import bs4

from .mm_abstract_loader import MMBaseLoader
from ..mm_types import MMDocument

default_header_template = {
    "User-Agent": "",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
    ";q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _build_metadata(soup: Any, url: str) -> dict:
    """Build metadata from BeautifulSoup output."""
    metadata = {"source": url}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", "No description found.")
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", "No language found.")
    return metadata


def _url_to_image(url: str) -> PILImageType:
    # thanks: https://stackoverflow.com/questions/12020657
    return Image.open(requests.get(url, stream=True).raw)


def _traverse(tree_base: bs4.BeautifulSoup) -> Iterable[dict]:
    """
    traverse the tree, escaping to *.text as soon as there
    are no 'img' in the subtree. Otherwise, go deeper and repeat.
    """
    if tree_base.name == "img":
        yield {"image_url": tree_base["src"]}
    else:
        if len(tree_base.find_all("img")) == 0:
            _txt = tree_base.text.strip()
            if _txt:
                yield {"text": _txt}
        else:
            for child in tree_base.children:
                if isinstance(child, bs4.element.NavigableString):
                    _txt = child.text.strip()
                    if _txt:
                        yield {"text": _txt}
                elif isinstance(child, bs4.element.Tag):
                    for piece in _traverse(child):
                        yield piece
                else:
                    raise ValueError(f"Unexpected child: {child}")


class MMWebBaseLoader(MMBaseLoader):

    def __init__(
        self,
        url: str,
    ) -> None:
        self.url = url
        session = requests.Session()
        session.headers = default_header_template
        self.session = session

    def _scrape(
        self,
        url: str,
    ) -> Any:
        parser = "html.parser"
        html_doc = self.session.get(url)
        return bs4.BeautifulSoup(html_doc.text, parser)

    def load(self) -> List[MMDocument]:
        soup = self._scrape(self.url)
        global_metadata = _build_metadata(soup, self.url)
        traversed = list(_traverse(soup))
        # combine adjacent texts
        documents = []
        buffer = []
        for trav in traversed:
            if "text" in trav and trav["text"]:
                buffer.append(trav["text"])
            elif "image_url" in trav:
                # flush buffer
                if buffer:
                    full_txt = "\n".join(buffer)
                    documents.append(MMDocument(content={"text": full_txt}, metadata=global_metadata))
                buffer = []
                # append image
                documents.append(MMDocument(
                    content={"image": _url_to_image(trav["image_url"])},
                    metadata={
                        **global_metadata,
                        **{"image_url": trav["image_url"]},
                    },
                ))
            else:
                raise ValueError(str(trav))
        # flush buffer
        if buffer:
            full_txt = "\n".join(buffer)
            documents.append(MMDocument(content={"text": full_txt}, metadata=global_metadata))
        buffer = []
        #
        return documents

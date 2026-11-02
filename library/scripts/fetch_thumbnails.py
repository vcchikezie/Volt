#!/usr/bin/env python3
"""Fetch a thumbnail image for each resource from its own web page.

    python3 scripts/fetch_thumbnails.py               # fetch missing thumbnails
    python3 scripts/fetch_thumbnails.py --refresh     # fetch all again
    python3 scripts/fetch_thumbnails.py --only LAW-001 CRT-004

For each resource in data/resources.json it opens the URL and takes, in order:
the page's preview image (og:image, twitter:image), else its largest icon
(apple-touch-icon, then other icons). Images are saved to data/thumbs/<ID>.<ext>
(resized to 640px wide when Pillow is installed), listed in data/thumbs/index.json,
and linked from resources.json, so no re-import is needed. The import script
keeps these links on later imports.

Needs internet access to the resources' sites. Pillow is optional
(pip install Pillow) and only used to shrink large images.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (compatible; VoltLibraryThumbnails/1.0; +https://github.com/vcchikezie/Volt)"
TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif", "image/svg+xml": "svg"}
MAX_HTML = 1_500_000
MAX_IMAGE = 5_000_000
WIDTH = 640

try:
    from PIL import Image
except ImportError:
    Image = None


class HeadParser(HTMLParser):
    """Collects preview-image meta tags and icon links."""

    def __init__(self):
        super().__init__()
        self.meta, self.icons = {}, []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key in ("og:image", "og:image:secure_url", "og:image:url", "twitter:image", "twitter:image:src") and a.get("content"):
                self.meta.setdefault(key, a["content"].strip())
        elif tag == "link" and a.get("href"):
            rel = a.get("rel", "").lower().split()
            if "apple-touch-icon" in rel or "apple-touch-icon-precomposed" in rel or "icon" in rel:
                size = 0
                for s in a.get("sizes", "").lower().split():
                    if "x" in s and s.split("x")[0].isdigit():
                        size = max(size, int(s.split("x")[0]))
                apple = any(r.startswith("apple-touch-icon") for r in rel)
                svg = a["href"].lower().endswith(".svg") or a.get("type") == "image/svg+xml"
                # Prefer big icons; an SVG scales, an apple-touch-icon is usually 180px.
                score = size or (512 if svg else 180 if apple else 32)
                self.icons.append((score, a["href"].strip()))


def get(url, limit, timeout):
    req = Request(url, headers={"User-Agent": UA, "Accept": "text/html,image/*;q=0.9,*/*;q=0.5"})
    with urlopen(req, timeout=timeout) as r:
        data = r.read(limit + 1)
        if len(data) > limit:
            raise ValueError("too large")
        return r.geturl(), r.headers.get_content_type(), data


def candidates(page_url, timeout):
    final, ctype, html = get(page_url, MAX_HTML, timeout)
    if ctype not in ("text/html", "application/xhtml+xml"):
        return []
    p = HeadParser()
    p.feed(html.decode("utf-8", "replace"))
    out = []
    for key in ("og:image:secure_url", "og:image", "og:image:url", "twitter:image", "twitter:image:src"):
        if key in p.meta:
            out.append(("photo", urljoin(final, p.meta[key])))
    for _, href in sorted(p.icons, reverse=True):
        out.append(("icon", urljoin(final, href)))
    out.append(("icon", urljoin(final, "/apple-touch-icon.png")))
    seen, uniq = set(), []
    for kind, u in out:
        if u.startswith(("http://", "https://")) and u not in seen:
            seen.add(u)
            uniq.append((kind, u))
    return uniq


def save_image(data, ctype, dest_base):
    ext = TYPES.get(ctype)
    if not ext:
        return None
    if Image and ext in ("jpg", "png", "webp"):
        try:
            im = Image.open(BytesIO(data))
            im.load()
            if im.width < 64 or im.height < 64:
                return None  # a tiny favicon looks worse than the generated cover
            if im.width > WIDTH:
                im = im.resize((WIDTH, round(im.height * WIDTH / im.width)))
            if ext == "jpg" or (im.mode in ("RGB", "L") and ext == "webp"):
                ext = "jpg"
                im = im.convert("RGB")
                buf = BytesIO()
                im.save(buf, "JPEG", quality=82, optimize=True)
            else:
                ext = "png"
                buf = BytesIO()
                im.save(buf, "PNG", optimize=True)
            data = buf.getvalue()
        except Exception:
            return None
    path = dest_base.with_suffix("." + ext)
    path.write_bytes(data)
    return path


def fetch_one(item, out_dir, timeout):
    rid, url = item["id"], item.get("url", "")
    if not url:
        return rid, None, "no link"
    try:
        cands = candidates(url, timeout)
    except (HTTPError, URLError, ValueError, TimeoutError, OSError) as e:
        return rid, None, f"page: {getattr(e, 'code', '') or e.__class__.__name__}"
    for kind, img_url in cands:
        try:
            _, ctype, data = get(img_url, MAX_IMAGE, timeout)
        except Exception:
            continue
        path = save_image(data, ctype, out_dir / rid)
        if path:
            return rid, {"file": f"data/thumbs/{path.name}", "kind": kind, "source": img_url, "fetched": date.today().isoformat()}, kind
    return rid, None, "no usable image"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="fetch every thumbnail again")
    ap.add_argument("--only", nargs="+", metavar="ID", help="only these resource IDs")
    ap.add_argument("--resources", type=Path, default=ROOT / "data" / "resources.json")
    ap.add_argument("--timeout", type=float, default=15)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    resources_path = args.resources
    out_dir = resources_path.parent / "thumbs"
    out_dir.mkdir(exist_ok=True)
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    data = json.loads(resources_path.read_text())
    todo = [r for r in data["resources"]
            if (not args.only or r["id"] in args.only) and (args.refresh or r["id"] not in index)]
    print(f"Fetching thumbnails for {len(todo)} resources…")

    found = {"photo": 0, "icon": 0}
    misses = []
    with ThreadPoolExecutor(args.workers) as pool:
        for rid, entry, note in pool.map(lambda r: fetch_one(r, out_dir, args.timeout), todo):
            if entry:
                old = index.get(rid)
                if old and old["file"] != entry["file"]:
                    (resources_path.parent.parent / old["file"]).unlink(missing_ok=True)
                index[rid] = entry
                found[entry["kind"]] += 1
            else:
                misses.append(f"{rid}: {note}")

    index_path.write_text(json.dumps(dict(sorted(index.items())), indent=2) + "\n")
    for r in data["resources"]:
        if r["id"] in index:
            r["thumb"] = {"src": index[r["id"]]["file"], "kind": index[r["id"]]["kind"]}
    resources_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    print(f"Preview images: {found['photo']}  Icons: {found['icon']}  Without an image: {len(misses)}")
    for m in misses:
        print(f"  - {m}")
    print(f"{len(index)} thumbnails in total. Commit data/thumbs/ and data/resources.json.")


if __name__ == "__main__":
    sys.exit(main())

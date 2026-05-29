#!/usr/bin/env python3
"""Fetch every page from nethackwiki.com as raw wikitext.

Uses the MediaWiki API (no scraping). Stdlib only (urllib, json).
Resumable: skips pages whose .txt file already exists on disk.
Rate-limited to 1 req/sec by default.

Output:
    data/wiki/pages/<safe_title>.txt   -- raw wikitext per page
    data/wiki/index.json               -- {title: {pageid, filename, categories, ns}}
    data/wiki/fetch.log                -- progress log

Usage:
    python scripts/fetch_wiki.py
    python scripts/fetch_wiki.py --delay 0.5 --batch 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://nethackwiki.com/mediawiki/api.php"
USER_AGENT = "AALL-LORE-WikiFetch/1.0 (research; contact: joconno2@conncoll.edu)"

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = REPO_ROOT / "data" / "wiki" / "pages"
INDEX_FILE = REPO_ROOT / "data" / "wiki" / "index.json"
LOG_FILE = REPO_ROOT / "data" / "wiki" / "fetch.log"

# Namespaces to fetch. 0 = main articles, 14 = categories.
# Skip Talk, User, User_talk, etc. Add more if needed.
NAMESPACES = [0, 14]

MAX_RETRIES = 5
RETRY_BACKOFF = 2.0  # seconds, doubled each retry


def setup_logging() -> logging.Logger:
    log = logging.getLogger("fetch_wiki")
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    return log


def safe_filename(title: str) -> str:
    """Convert page title to a filesystem-safe filename.

    Replaces /, :, and other problematic chars with underscores.
    Truncates to 200 chars to stay under filesystem limits.
    """
    name = title.replace("/", "_SLASH_").replace(":", "_COLON_")
    name = re.sub(r'[<>"|?*\\]', "_", name)
    name = re.sub(r"\s+", "_", name)
    if len(name) > 200:
        name = name[:200]
    return name + ".txt"


def api_request(params: dict, delay: float, log: logging.Logger) -> dict:
    """Make a GET request to the MediaWiki API with retries."""
    params["format"] = "json"
    url = API_URL + "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("Accept", "application/json")

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(delay)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            log.warning("API error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, MAX_RETRIES, e, wait)
            time.sleep(wait)

    log.error("Failed after %d retries: %s", MAX_RETRIES, url)
    raise RuntimeError(f"API request failed after {MAX_RETRIES} retries")


def enumerate_all_pages(ns: int, delay: float, log: logging.Logger) -> list[dict]:
    """Get all page titles + pageids in a namespace via allpages."""
    pages = []
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": str(ns),
        "aplimit": "500",
    }
    batch = 0
    while True:
        batch += 1
        data = api_request(params, delay, log)
        batch_pages = data.get("query", {}).get("allpages", [])
        pages.extend(batch_pages)
        log.info("NS %d batch %d: got %d pages (total %d)", ns, batch, len(batch_pages), len(pages))

        if "continue" in data:
            params["apcontinue"] = data["continue"]["apcontinue"]
        else:
            break

    return pages


def fetch_page_content(pageid: int, delay: float, log: logging.Logger) -> tuple[str, list[str]]:
    """Fetch wikitext content and categories for a single page by pageid.

    Returns (wikitext, [category_titles]).
    """
    params = {
        "action": "query",
        "pageids": str(pageid),
        "prop": "revisions|categories",
        "rvprop": "content",
        "rvslots": "main",
        "cllimit": "500",
    }
    data = api_request(params, delay, log)
    page = data.get("query", {}).get("pages", {}).get(str(pageid), {})

    # Extract wikitext
    revisions = page.get("revisions", [])
    if revisions:
        slots = revisions[0].get("slots", {})
        content = slots.get("main", {}).get("*", "")
    else:
        content = ""

    # Extract categories
    cats = [c["title"] for c in page.get("categories", [])]

    return content, cats


def load_index() -> dict:
    """Load existing index from disk, or return empty dict."""
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_index(index: dict) -> None:
    """Atomic write of index JSON."""
    tmp = INDEX_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    tmp.rename(INDEX_FILE)


def main():
    parser = argparse.ArgumentParser(description="Fetch NetHack wiki pages")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API requests (default: 1.0)")
    parser.add_argument("--batch", type=int, default=0,
                        help="Stop after N pages fetched this run (0 = all)")
    parser.add_argument("--ns", type=int, nargs="+", default=NAMESPACES,
                        help="Namespace IDs to fetch (default: 0 14)")
    args = parser.parse_args()

    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging()
    log.info("=== fetch_wiki.py starting ===")
    log.info("Output dir: %s", PAGES_DIR)
    log.info("Delay: %.1fs, batch limit: %s, namespaces: %s",
             args.delay, args.batch or "unlimited", args.ns)

    # Phase 1: enumerate all pages
    all_pages = []
    for ns in args.ns:
        log.info("Enumerating namespace %d...", ns)
        ns_pages = enumerate_all_pages(ns, args.delay, log)
        for p in ns_pages:
            p["ns"] = ns
        all_pages.extend(ns_pages)

    log.info("Total pages across all namespaces: %d", len(all_pages))

    # Phase 2: load existing index, figure out what to skip
    index = load_index()
    existing_files = set(os.listdir(PAGES_DIR))

    fetched = 0
    skipped = 0
    errors = 0

    for i, page in enumerate(all_pages):
        title = page["title"]
        pageid = page["pageid"]
        ns = page["ns"]
        fname = safe_filename(title)

        # Skip if already downloaded
        if fname in existing_files:
            skipped += 1
            continue

        # Fetch content
        try:
            content, categories = fetch_page_content(pageid, args.delay, log)
        except RuntimeError:
            log.error("Skipping page %s (pageid=%d) after repeated failures", title, pageid)
            errors += 1
            continue

        # Write page file
        page_path = PAGES_DIR / fname
        tmp_path = page_path.with_suffix(".txt.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        tmp_path.rename(page_path)

        # Update index
        index[title] = {
            "pageid": pageid,
            "filename": fname,
            "categories": categories,
            "ns": ns,
        }

        fetched += 1
        if fetched % 50 == 0:
            save_index(index)
            log.info("Progress: %d fetched, %d skipped, %d errors (page %d/%d)",
                     fetched, skipped, errors, i + 1, len(all_pages))

        if args.batch and fetched >= args.batch:
            log.info("Batch limit %d reached, stopping", args.batch)
            break

    # Final save
    save_index(index)
    log.info("=== Done. Fetched: %d, skipped: %d, errors: %d, index entries: %d ===",
             fetched, skipped, errors, len(index))


if __name__ == "__main__":
    main()

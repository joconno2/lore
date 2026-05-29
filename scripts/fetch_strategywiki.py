#!/usr/bin/env python3
"""Fetch every NetHack page from StrategyWiki as raw wikitext.

Uses the MediaWiki API at strategywiki.org. Stdlib only (urllib, json).
Resumable: skips pages whose .txt file already exists on disk.
Rate-limited to 1 req/sec by default.

Discovery strategy: recursively walks Category:NetHack and all subcategories
via action=query&list=categorymembers. Also fetches the main NetHack page
itself (which may not be in the category).

Output:
    data/guides/strategywiki/<safe_title>.txt   -- raw wikitext per page
    data/guides/strategywiki/index.json         -- {title: {pageid, filename, ns}}
    data/guides/strategywiki/fetch.log          -- progress log

Usage:
    python scripts/fetch_strategywiki.py
    python scripts/fetch_strategywiki.py --delay 0.5 --batch 50
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

API_URL = "https://strategywiki.org/w/api.php"
USER_AGENT = "AALL-LORE-StrategyWikiFetch/1.0 (research; contact: joconno2@conncoll.edu)"

ROOT_CATEGORY = "Category:NetHack"
# Seed pages that may not appear in the category tree.
SEED_TITLES = ["NetHack"]

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "guides" / "strategywiki"
INDEX_FILE = OUT_DIR / "index.json"
LOG_FILE = OUT_DIR / "fetch.log"

MAX_RETRIES = 5
RETRY_BACKOFF = 2.0


def setup_logging() -> logging.Logger:
    log = logging.getLogger("fetch_strategywiki")
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
    """Convert page title to a filesystem-safe filename."""
    name = title.replace("/", "_SLASH_").replace(":", "_COLON_")
    name = re.sub(r'[<>"|?*\\]', "_", name)
    name = re.sub(r"\s+", "_", name)
    if len(name) > 200:
        name = name[:200]
    return name + ".txt"


def api_request(params: dict, delay: float, log: logging.Logger) -> dict:
    """Make a GET request to the StrategyWiki API with retries."""
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


def enumerate_category(category: str, delay: float, log: logging.Logger,
                       seen_cats: set[str] | None = None) -> list[dict]:
    """Recursively enumerate all pages under a category.

    Returns list of dicts with keys: title, pageid, ns.
    Walks subcategories (ns=14) recursively. Tracks visited categories
    to avoid cycles.
    """
    if seen_cats is None:
        seen_cats = set()

    if category in seen_cats:
        return []
    seen_cats.add(category)

    pages = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": "500",
        "cmprop": "ids|title|type",
    }

    batch = 0
    while True:
        batch += 1
        data = api_request(params, delay, log)
        members = data.get("query", {}).get("categorymembers", [])
        log.info("Category %s batch %d: %d members", category, batch, len(members))

        for m in members:
            if m["type"] == "subcat":
                # Recurse into subcategory
                sub_pages = enumerate_category(m["title"], delay, log, seen_cats)
                pages.extend(sub_pages)
            else:
                pages.append({
                    "title": m["title"],
                    "pageid": m["pageid"],
                    "ns": m.get("ns", 0),
                })

        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
        else:
            break

    return pages


def resolve_title(title: str, delay: float, log: logging.Logger) -> dict | None:
    """Resolve a page title to pageid via the API. Returns None if missing."""
    params = {
        "action": "query",
        "titles": title,
    }
    data = api_request(params, delay, log)
    query_pages = data.get("query", {}).get("pages", {})
    for pid, info in query_pages.items():
        if int(pid) > 0:
            return {"title": info["title"], "pageid": info["pageid"], "ns": info.get("ns", 0)}
    return None


def fetch_page_content(pageid: int, delay: float, log: logging.Logger) -> str:
    """Fetch wikitext for a single page by pageid."""
    params = {
        "action": "query",
        "pageids": str(pageid),
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
    }
    data = api_request(params, delay, log)
    page = data.get("query", {}).get("pages", {}).get(str(pageid), {})

    revisions = page.get("revisions", [])
    if revisions:
        slots = revisions[0].get("slots", {})
        content = slots.get("main", {}).get("*", "")
    else:
        content = ""

    return content


def load_index() -> dict:
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_index(index: dict) -> None:
    tmp = INDEX_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    tmp.rename(INDEX_FILE)


def main():
    parser = argparse.ArgumentParser(description="Fetch StrategyWiki NetHack pages")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API requests (default: 1.0)")
    parser.add_argument("--batch", type=int, default=0,
                        help="Stop after N pages fetched this run (0 = all)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging()
    log.info("=== fetch_strategywiki.py starting ===")
    log.info("Output dir: %s", OUT_DIR)
    log.info("Delay: %.1fs, batch limit: %s", args.delay, args.batch or "unlimited")

    # Phase 1: discover all NetHack pages via category tree
    log.info("Walking category tree from %s...", ROOT_CATEGORY)
    all_pages = enumerate_category(ROOT_CATEGORY, args.delay, log)

    # Add seed pages that might not be categorized
    seen_ids = {p["pageid"] for p in all_pages}
    for title in SEED_TITLES:
        info = resolve_title(title, args.delay, log)
        if info and info["pageid"] not in seen_ids:
            all_pages.append(info)
            seen_ids.add(info["pageid"])
            log.info("Added seed page: %s (pageid=%d)", info["title"], info["pageid"])

    # Also try prefix-based discovery for pages like "NetHack/Foo" not in the category
    log.info("Checking for prefix-based pages (NetHack/*)...")
    prefix_pages = enumerate_by_prefix("NetHack/", args.delay, log)
    for p in prefix_pages:
        if p["pageid"] not in seen_ids:
            all_pages.append(p)
            seen_ids.add(p["pageid"])
    log.info("After prefix scan: %d total unique pages", len(all_pages))

    log.info("Total pages discovered: %d", len(all_pages))

    # Phase 2: fetch content for each page
    index = load_index()
    existing_files = set(os.listdir(OUT_DIR))

    fetched = 0
    skipped = 0
    errors = 0

    for i, page in enumerate(all_pages):
        title = page["title"]
        pageid = page["pageid"]
        fname = safe_filename(title)

        if fname in existing_files:
            skipped += 1
            continue

        try:
            content = fetch_page_content(pageid, args.delay, log)
        except RuntimeError:
            log.error("Skipping %s (pageid=%d) after repeated failures", title, pageid)
            errors += 1
            continue

        page_path = OUT_DIR / fname
        tmp_path = page_path.with_suffix(".txt.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        tmp_path.rename(page_path)

        index[title] = {
            "pageid": pageid,
            "filename": fname,
            "ns": page.get("ns", 0),
        }

        fetched += 1
        if fetched % 20 == 0:
            save_index(index)
            log.info("Progress: %d fetched, %d skipped, %d errors (page %d/%d)",
                     fetched, skipped, errors, i + 1, len(all_pages))

        if args.batch and fetched >= args.batch:
            log.info("Batch limit %d reached, stopping", args.batch)
            break

    save_index(index)
    log.info("=== Done. Fetched: %d, skipped: %d, errors: %d, index entries: %d ===",
             fetched, skipped, errors, len(index))


def enumerate_by_prefix(prefix: str, delay: float, log: logging.Logger) -> list[dict]:
    """Find all pages starting with a given prefix via allpages."""
    pages = []
    params = {
        "action": "query",
        "list": "allpages",
        "apprefix": prefix,
        "apnamespace": "0",
        "aplimit": "500",
    }
    batch = 0
    while True:
        batch += 1
        data = api_request(params, delay, log)
        batch_pages = data.get("query", {}).get("allpages", [])
        for p in batch_pages:
            pages.append({"title": p["title"], "pageid": p["pageid"], "ns": 0})
        log.info("Prefix '%s' batch %d: %d pages (total %d)", prefix, batch, len(batch_pages), len(pages))

        if "continue" in data:
            params["apcontinue"] = data["continue"]["apcontinue"]
        else:
            break

    return pages


if __name__ == "__main__":
    main()

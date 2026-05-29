#!/usr/bin/env python3
"""Parse NetHackWiki XML dump into individual page text files + index JSON.

Iterative parsing via xml.etree.ElementTree.iterparse. Stdlib only.
Keeps namespace 0 (main articles) and namespace 14 (categories).
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET

WIKI_XML = "/home/jim/code/lore/data/wiki/nethackwiki_current.xml"
PAGES_DIR = "/home/jim/code/lore/data/wiki/pages"
INDEX_PATH = "/home/jim/code/lore/data/wiki/index.json"

# MediaWiki export namespace (0.10 for this dump)
MW_NS = "http://www.mediawiki.org/xml/export-0.10/"

KEEP_NAMESPACES = {0, 14}

# Unsafe filename characters
UNSAFE_RE = re.compile(r'[<>:"|?*\x00-\x1f]')


def tag(local):
    """Return fully qualified tag name."""
    return f"{{{MW_NS}}}{local}"


def safe_filename(title):
    """Convert page title to a safe filename."""
    name = title.replace("/", "_")
    name = UNSAFE_RE.sub("", name)
    # Collapse runs of whitespace
    name = re.sub(r"\s+", " ", name).strip()
    # Truncate to avoid filesystem limits (leave room for .txt)
    if len(name.encode("utf-8")) > 240:
        name = name[:200]
    return name + ".txt"


def parse_dump():
    os.makedirs(PAGES_DIR, exist_ok=True)

    index = {}
    kept = 0
    skipped = 0
    total = 0

    context = ET.iterparse(WIKI_XML, events=("end",))

    for event, elem in context:
        if elem.tag != tag("page"):
            continue

        total += 1

        title_el = elem.find(tag("title"))
        ns_el = elem.find(tag("ns"))
        id_el = elem.find(tag("id"))

        title = title_el.text if title_el is not None else ""
        ns = int(ns_el.text) if ns_el is not None and ns_el.text else -1
        pageid = int(id_el.text) if id_el is not None and id_el.text else 0

        if ns not in KEEP_NAMESPACES:
            skipped += 1
            elem.clear()
            continue

        # Get latest revision text (one revision per page in this dump)
        revision = elem.find(tag("revision"))
        text_el = revision.find(tag("text")) if revision is not None else None
        wikitext = text_el.text or "" if text_el is not None else ""

        fname = safe_filename(title)
        fpath = os.path.join(PAGES_DIR, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(wikitext)

        index[title] = {
            "filename": fname,
            "namespace": ns,
            "pageid": pageid,
        }

        kept += 1

        if total % 1000 == 0:
            print(f"  processed {total} pages, kept {kept}, skipped {skipped}")

        # Free memory
        elem.clear()

    print(f"Done. {total} pages total, {kept} kept, {skipped} skipped.")

    # Write index atomically
    tmp_path = INDEX_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    os.replace(tmp_path, INDEX_PATH)
    print(f"Index written to {INDEX_PATH} ({len(index)} entries)")


if __name__ == "__main__":
    parse_dump()

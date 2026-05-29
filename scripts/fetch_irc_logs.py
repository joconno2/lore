#!/usr/bin/env python3
"""Fetch Hardfought IRC logs for #nethack and #hardfought channels.

Scrapes the channel index pages at hardfought.org/nethack/irclogs/<channel>/
and downloads each .log file. Stdlib only (urllib, html.parser).

Resumable: skips files that already exist on disk.
Rate-limited to 1 req/sec by default.

Output:
    data/irc/hardfought/<channel>/<logfile>.log

Usage:
    python scripts/fetch_irc_logs.py
    python scripts/fetch_irc_logs.py --delay 0.5
    python scripts/fetch_irc_logs.py --channels hardfought nethack tnnt
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE_URL = "https://www.hardfought.org/nethack/irclogs/"
USER_AGENT = "AALL-LORE-IRCFetch/1.0 (research; contact: joconno2@conncoll.edu)"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "irc" / "hardfought"
LOG_FILE = REPO_ROOT / "data" / "irc" / "fetch_irc.log"

DEFAULT_CHANNELS = ["nethack", "hardfought"]

MAX_RETRIES = 5
RETRY_BACKOFF = 2.0


class LinkParser(HTMLParser):
    """Extract href values from <a> tags."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)


def setup_logging() -> logging.Logger:
    log = logging.getLogger("fetch_irc_logs")
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    return log


def fetch_url(url: str, delay: float, log: logging.Logger) -> bytes | None:
    """GET a URL with retries and rate limiting. Returns bytes or None on failure."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(delay)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            log.warning("Fetch error (attempt %d/%d) %s: %s. Retry in %.1fs",
                        attempt + 1, MAX_RETRIES, url, e, wait)
            time.sleep(wait)

    log.error("Failed after %d retries: %s", MAX_RETRIES, url)
    return None


def parse_log_links(html: str, channel: str) -> list[str]:
    """Extract .log filenames from a channel index page."""
    parser = LinkParser()
    parser.feed(html)
    logs = []
    for link in parser.links:
        # Links may be relative (just filename) or include the channel path.
        # Normalize to just the filename.
        basename = link.rsplit("/", 1)[-1] if "/" in link else link
        if basename.endswith(".log"):
            logs.append(basename)
    return logs


def fetch_channel(channel: str, delay: float, log: logging.Logger) -> tuple[int, int, int]:
    """Fetch all logs for one channel. Returns (fetched, skipped, errors)."""
    channel_url = BASE_URL + channel + "/"
    log.info("Fetching channel index: %s", channel_url)

    html_bytes = fetch_url(channel_url, delay, log)
    if html_bytes is None:
        log.error("Could not fetch channel index for %s, skipping", channel)
        return 0, 0, 1

    html = html_bytes.decode("utf-8", errors="replace")
    log_files = parse_log_links(html, channel)
    log.info("Channel %s: found %d log files", channel, len(log_files))

    if not log_files:
        return 0, 0, 0

    out_dir = OUTPUT_DIR / channel
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = set(os.listdir(out_dir))

    fetched = 0
    skipped = 0
    errors = 0

    for i, filename in enumerate(log_files):
        if filename in existing:
            skipped += 1
            continue

        file_url = channel_url + urllib.parse.quote(filename)
        data = fetch_url(file_url, delay, log)

        if data is None:
            log.error("Failed to download %s/%s", channel, filename)
            errors += 1
            continue

        # Atomic write
        dest = out_dir / filename
        tmp = dest.with_suffix(".log.tmp")
        tmp.write_bytes(data)
        tmp.rename(dest)
        fetched += 1

        if fetched % 50 == 0:
            log.info("  %s progress: %d fetched, %d skipped, %d errors (file %d/%d)",
                     channel, fetched, skipped, errors, i + 1, len(log_files))

    return fetched, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="Fetch Hardfought IRC logs")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between requests (default: 1.0)")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS,
                        help="Channel names to fetch (default: nethack hardfought)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging()
    log.info("=== fetch_irc_logs.py starting ===")
    log.info("Output dir: %s", OUTPUT_DIR)
    log.info("Delay: %.1fs, channels: %s", args.delay, args.channels)

    total_fetched = 0
    total_skipped = 0
    total_errors = 0

    for channel in args.channels:
        fetched, skipped, errors = fetch_channel(channel, args.delay, log)
        log.info("Channel %s done: %d fetched, %d skipped, %d errors",
                 channel, fetched, skipped, errors)
        total_fetched += fetched
        total_skipped += skipped
        total_errors += errors

    log.info("=== Done. Fetched: %d, skipped: %d, errors: %d ===",
             total_fetched, total_skipped, total_errors)


if __name__ == "__main__":
    main()

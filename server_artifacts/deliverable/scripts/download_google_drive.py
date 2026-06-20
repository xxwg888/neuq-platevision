#!/usr/bin/env python
"""Download public Google Drive files without installing gdown."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from urllib.parse import urljoin

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    return parser.parse_args()


def get_confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
    if match:
        return match.group(1)
    return None


def get_confirm_url(response: requests.Response) -> str | None:
    text = response.text
    patterns = [
        r'href="(/uc\?export=download[^"]+)"',
        r'action="([^"]*/uc\?export=download[^"]*)"',
        r'href="(https://drive\.usercontent\.google\.com/download[^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return html.unescape(urljoin("https://docs.google.com", match.group(1)))
    return None


def get_confirm_form(response: requests.Response) -> tuple[str, dict[str, str]] | None:
    text = response.text
    action_match = re.search(r'action="([^"]+)"', text)
    if not action_match:
        return None
    fields = {
        html.unescape(name): html.unescape(value)
        for name, value in re.findall(r'name="([^"]+)" value="([^"]*)"', text)
    }
    if not fields:
        return None
    return html.unescape(action_match.group(1)), fields


def save_response(response: requests.Response, output: Path, chunk_size: int):
    total = int(response.headers.get("Content-Length", 0))
    done = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in response.iter_content(chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 / total
                print(f"\r{done / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r{done / 1024 / 1024:.1f} MB", end="", flush=True)
    print()
    tmp.replace(output)


def main():
    args = parse_args()
    output = Path(args.output)
    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    response = session.get(url, params={"id": args.file_id}, stream=True, timeout=60)
    token = get_confirm_token(response)
    if token:
        response.close()
        response = session.get(url, params={"id": args.file_id, "confirm": token}, stream=True, timeout=60)
    elif "text/html" in response.headers.get("Content-Type", ""):
        confirm_url = get_confirm_url(response)
        if confirm_url:
            response.close()
            response = session.get(confirm_url, stream=True, timeout=60)
        else:
            form = get_confirm_form(response)
            if form:
                action, fields = form
                response.close()
                response = session.get(action, params=fields, stream=True, timeout=60)
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        snippet = response.text[:500]
        raise RuntimeError(f"Google Drive returned HTML instead of file content: {snippet}")
    save_response(response, output, args.chunk_size)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()

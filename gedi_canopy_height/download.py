import os
import posixpath
from typing import Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT: Tuple[float, float] = (10.0, 60.0)


def _parse_content_range(content_range: str) -> tuple[int, int, int] | None:
    # Example: "bytes 100-199/1000"
    try:
        units, values = content_range.split(" ", 1)
        if units.lower() != "bytes":
            return None
        byte_range, total = values.split("/")
        start_text, end_text = byte_range.split("-")
        return int(start_text), int(end_text), int(total)
    except (ValueError, AttributeError):
        return None


def _build_session(max_retries: int = 5) -> requests.Session:
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def download(
    URL: str,
    filename: str,
    timeout: Tuple[float, float] = DEFAULT_TIMEOUT,
    chunk_size: int = 1024 * 1024,
    session: Optional[requests.Session] = None,
) -> None:
    # Get the current file size if it exists.
    existing_file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
    # Request the file details using Range header if a partial file exists.
    headers = {"Range": f"bytes={existing_file_size}-"} if existing_file_size else {}
    own_session = session is None
    session = session or _build_session()

    try:
        response = session.get(URL, headers=headers, stream=True, timeout=timeout)

        # If server ignores range requests, restart from scratch.
        if existing_file_size and response.status_code == 200:
            existing_file_size = 0
            headers = {}
            response = session.get(URL, headers=headers, stream=True, timeout=timeout)

        # Check for resumable support (206 Partial Content response).
        if response.status_code not in (200, 206):
            raise IOError(f"download failed for {URL}: HTTP {response.status_code}")

        # If server returns a mismatched range, restart from scratch to avoid file corruption.
        if existing_file_size and response.status_code == 206:
            parsed = _parse_content_range(response.headers.get("Content-Range", ""))
            if parsed is None or parsed[0] != existing_file_size:
                response.close()
                existing_file_size = 0
                response = session.get(URL, headers={}, stream=True, timeout=timeout)
                if response.status_code != 200:
                    raise IOError(
                        "download resume mismatch and restart failed "
                        f"for {URL}: HTTP {response.status_code}"
                    )

        # Determine total size based on Content-Range or Content-Length.
        content_range = response.headers.get("Content-Range")
        if content_range:
            total_size = int(content_range.split("/")[-1])
        else:
            total_size = int(response.headers.get("Content-Length", 0)) + existing_file_size

        # Set mode to append if resuming, otherwise write.
        mode = "ab" if existing_file_size else "wb"

        # Start download.
        with open(filename, mode) as f, tqdm(
            initial=existing_file_size,
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {posixpath.basename(URL)}",
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        if total_size > 0:
            downloaded_size = os.path.getsize(filename)
            if downloaded_size != total_size:
                raise IOError(
                    f"downloaded size mismatch for {URL}: expected {total_size}, got {downloaded_size}"
                )
    finally:
        if own_session:
            session.close()

    print(f"Download completed: {filename}")
"""A seekable file object backed by HTTP range requests.

`Dataset.zip` is 9.34 GiB and unzips to 14.00 GiB, which does not fit in this
project's 1 GB disk budget. But a zip's central directory lives at the end of
the file and records every member's offset and compressed size, so given a
*seekable* file object Python's own `zipfile` can pull out one member without
ever seeing the rest. `HttpFile` is that object.

    with HttpFile(DATASET_URL) as fh:
        zf = zipfile.ZipFile(fh)
        data = zf.read("Dataset/<sim>/<sim>_aerofoil.vtp")   # fetches ~0.3 MB

This also turns out to be *more* robust than downloading the archive in one
stream: each member is an independent, individually retryable request, so a
dropped connection costs one simulation rather than restarting 9.34 GiB.
`zipfile` verifies each member's CRC on extraction, which gives us per-member
integrity checking for free.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import requests

#: Read-ahead window for small reads. Central-directory parsing does many
#: small reads; serving them from one window keeps the request count sane.
CHUNK = 1 << 20  # 1 MiB

#: The tail is read repeatedly while locating the end-of-central-directory
#: record (twice over for zip64), so it is fetched once and kept.
TAIL_BYTES = 2 << 20  # 2 MiB


class RangeNotSupported(RuntimeError):
    """The server will not serve partial content, so streaming is impossible."""


class HttpFile(io.RawIOBase):
    """Read-only, seekable view of a remote file via `Range` requests."""

    def __init__(
        self,
        url: str,
        session: Optional[requests.Session] = None,
        timeout: float = 60.0,
        retries: int = 6,
        expected_size: Optional[int] = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._pos = 0
        self.bytes_fetched = 0
        self.requests_made = 0

        # The host is intermittently unreachable (airfrans_lib issue #6 is an
        # unfixed connection timeout against it), so even the opening HEAD gets
        # the retry treatment -- a transient blip here would otherwise abort a
        # run that is perfectly capable of continuing.
        head = self._with_retries(
            lambda: self._session.head(url, timeout=timeout, allow_redirects=True),
            what="HEAD",
        )
        head.raise_for_status()
        if head.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RangeNotSupported(
                f"{url} does not advertise 'Accept-Ranges: bytes'; "
                "streaming individual members is not possible."
            )
        self._size = int(head.headers["Content-Length"])
        if expected_size is not None and self._size != expected_size:
            raise RuntimeError(
                f"{url} is {self._size} bytes, expected {expected_size}. "
                "The dataset may have been republished; re-verify before trusting results."
            )

        self._win = b""        # small-read window
        self._win_start = -1
        self._tail = b""       # persistent cache of the last TAIL_BYTES
        self._tail_start = -1

    # -- file protocol ----------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(0, min(new, self._size))
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        n = min(n, self._size - self._pos)
        if n <= 0:
            return b""

        data = self._from_tail(self._pos, n)
        if data is None:
            if n >= CHUNK:
                # Large sequential read (a member's payload): one request, and
                # deliberately not cached -- we stream each member exactly once.
                data = self._fetch(self._pos, self._pos + n)
            else:
                data = self._from_window(self._pos, n)

        self._pos += len(data)
        return data

    def readinto(self, b) -> int:  # type: ignore[override]
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self) -> None:
        if not self.closed and self._owns_session:
            self._session.close()
        super().close()

    # -- caching ----------------------------------------------------------

    def _from_tail(self, start: int, n: int) -> Optional[bytes]:
        """Serve from the cached tail, fetching it on first use near the end."""
        tail_begin = max(0, self._size - TAIL_BYTES)
        if start < tail_begin:
            return None
        if self._tail_start < 0:
            self._tail = self._fetch(tail_begin, self._size)
            self._tail_start = tail_begin
        off = start - self._tail_start
        return self._tail[off : off + n]

    def _from_window(self, start: int, n: int) -> bytes:
        if not (0 <= self._win_start <= start and start + n <= self._win_start + len(self._win)):
            end = min(start + CHUNK, self._size)
            self._win = self._fetch(start, end)
            self._win_start = start
        off = start - self._win_start
        return self._win[off : off + n]

    # -- transport --------------------------------------------------------

    def _with_retries(self, call, what: str):
        """Run `call`, retrying any transport failure with exponential backoff.

        `RangeNotSupported` is never retried -- it is a statement about the
        server's capabilities, not a transient fault.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                return call()
            except RangeNotSupported:
                raise
            except Exception as exc:  # noqa: BLE001 - retry any transport failure
                last_exc = exc
                if attempt < self.retries - 1:
                    time.sleep(min(2.0**attempt, 30.0))
        raise OSError(
            f"{what} on {self.url} failed after {self.retries} attempts"
        ) from last_exc

    def _fetch(self, start: int, end: int) -> bytes:
        """GET bytes [start, end) with retries. `end` is exclusive."""
        headers = {"Range": f"bytes={start}-{end - 1}"}

        def once() -> bytes:
            resp = self._session.get(
                self.url, headers=headers, timeout=self.timeout, stream=True
            )
            try:
                if resp.status_code != 206:
                    # A 200 means the server ignored Range and is about to send
                    # the whole 9.34 GiB. Abort rather than silently take it.
                    raise RangeNotSupported(
                        f"expected 206 Partial Content, got {resp.status_code}"
                    )
                payload = resp.content
            finally:
                resp.close()
            if len(payload) != end - start:
                raise OSError(
                    f"short range response: asked {end - start} bytes, got {len(payload)}"
                )
            return payload

        payload = self._with_retries(once, what=f"GET bytes {start}-{end}")
        self.bytes_fetched += len(payload)
        self.requests_made += 1
        return payload

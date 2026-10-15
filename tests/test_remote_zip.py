"""The HTTP range-request file object that makes streaming the dataset possible."""

import io
import re
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from remote_zip import HttpFile, RangeNotSupported

PAYLOAD: bytes = b""
SUPPORT_RANGES = True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_HEAD(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        if SUPPORT_RANGES:
            self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d+)-(\d+)", rng or "")
        if not m or not SUPPORT_RANGES:
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
            return
        start, end = int(m.group(1)), int(m.group(2))
        chunk = PAYLOAD[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/archive.zip"
    srv.shutdown()


@pytest.fixture
def archive():
    """A zip bigger than the tail cache, so partial reads are actually partial.

    Stored (uncompressed) random bytes: a repetitive payload would deflate to a
    couple of kilobytes, the whole archive would fit inside the 2 MiB tail
    cache, and the test would prove nothing.
    """
    global PAYLOAD
    rng = __import__("random").Random(0)
    members = {
        f"Dataset/sim{i}/sim{i}_internal.vtu": bytes(rng.getrandbits(8) for _ in range(600_000))
        for i in range(5)
    }
    members["Dataset/manifest.json"] = b'{"full_test": ["a"], "scarce_train": ["b"]}'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    PAYLOAD = buf.getvalue()
    assert len(PAYLOAD) > 2 << 20
    return members


def test_reads_members_without_downloading_everything(server, archive):
    with HttpFile(server) as fh:
        zf = zipfile.ZipFile(fh)
        assert set(zf.namelist()) == set(archive)
        assert zf.read("Dataset/manifest.json") == archive["Dataset/manifest.json"]
        assert zf.read("Dataset/sim2/sim2_internal.vtu") == \
            archive["Dataset/sim2/sim2_internal.vtu"]
        # The whole point: one member plus the directory, not the whole archive.
        assert fh.bytes_fetched < len(PAYLOAD)


def test_seek_and_tell_behave(server, archive):
    with HttpFile(server) as fh:
        assert fh.seekable() and fh.readable() and not fh.writable()
        assert fh.size == len(PAYLOAD)
        fh.seek(10)
        assert fh.tell() == 10
        assert fh.read(4) == PAYLOAD[10:14]
        fh.seek(-6, io.SEEK_END)
        assert fh.read() == PAYLOAD[-6:]
        fh.seek(0)
        assert fh.read(3) == PAYLOAD[:3]


def test_size_mismatch_is_refused(server, archive):
    with pytest.raises(RuntimeError, match="expected"):
        HttpFile(server, expected_size=len(PAYLOAD) + 1)


def test_server_without_range_support_is_refused(server, archive):
    """Rather than silently downloading 9.34 GiB into a 1 GB disk budget."""
    global SUPPORT_RANGES
    SUPPORT_RANGES = False
    try:
        with pytest.raises(RangeNotSupported):
            HttpFile(server)
    finally:
        SUPPORT_RANGES = True

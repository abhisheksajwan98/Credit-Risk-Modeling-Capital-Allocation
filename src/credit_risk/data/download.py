"""Fetch the raw LendingClub extract from Kaggle, or generate a synthetic stand-in.

Uses the Kaggle REST API through the standard library rather than the ``kaggle`` CLI, for three
reasons: it avoids a dependency that shells out and calls ``sys.exit`` on failure, it allows
downloading a *single file* instead of the whole 1.36 GB archive, and the whole auth path stays
visible in twenty lines instead of behind a package.

Credentials are read from ``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``
environment variables. The key is never logged, never written to a manifest, and never included
in an error message.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from credit_risk.utils.runtime import get_logger

LOG = get_logger("data.download")

KAGGLE_DATASET = "wordsforthewise/lending-club"
KAGGLE_API_ROOT = "https://www.kaggle.com/api/v1"

ACCEPTED_FILE = "accepted_2007_to_2018Q4.csv.gz"
REJECTED_FILE = "rejected_2007_to_2018Q4.csv.gz"

CHECKSUM_FILE = "checksums.json"


class KaggleAuthError(RuntimeError):
    """Raised when credentials are missing or rejected, with instructions rather than a traceback."""


@dataclass(frozen=True)
class KaggleCredentials:
    username: str
    key: str

    def auth_header(self) -> str:
        if self.key.startswith("KGAT_"):
            return f"Bearer {self.key}"
        token = base64.b64encode(f"{self.username}:{self.key}".encode()).decode("ascii")
        return f"Basic {token}"

    def __repr__(self) -> str:  # pragma: no cover - never leak the key
        return f"KaggleCredentials(username={self.username!r}, key=<redacted>)"


def load_credentials() -> KaggleCredentials:
    """Resolve Kaggle credentials, preferring environment variables over the JSON file."""
    env_token = os.environ.get("KAGGLE_API_TOKEN")
    if env_token:
        return KaggleCredentials("", env_token)

    env_user, env_key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return KaggleCredentials(env_user, env_key)

    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return KaggleCredentials("", token)

    for candidate in (
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json"
        if os.environ.get("KAGGLE_CONFIG_DIR")
        else None,
        Path.home() / ".kaggle" / "kaggle.json",
    ):
        if candidate and candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                return KaggleCredentials(payload["username"], payload["key"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise KaggleAuthError(
                    f"{candidate} exists but is not a valid Kaggle token file "
                    f"(expected keys 'username' and 'key')."
                ) from exc

    raise KaggleAuthError(
        "No Kaggle credentials found.\n"
        "  1. Sign in at https://www.kaggle.com, open Settings -> API -> Create New Token.\n"
        f"  2. Save the downloaded kaggle.json to {Path.home() / '.kaggle' / 'kaggle.json'},\n"
        f"     or save the API token string to {Path.home() / '.kaggle' / 'access_token'}.\n"
        "  Alternatively set KAGGLE_API_TOKEN, or KAGGLE_USERNAME/KAGGLE_KEY in the environment.\n"
        "  No credentials yet? Run with --synthetic to generate a schema-faithful stand-in."
    )


def sha256sum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, credentials: KaggleCredentials, expect_zip: bool) -> Path:
    """Stream a URL to ``dest``, reporting progress on a single rewritten line."""
    request = urllib.request.Request(url, headers={"Authorization": credentials.auth_header()})
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with tmp.open("wb") as out:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if total and sys.stderr.isatty():
                        pct = 100.0 * done / total
                        sys.stderr.write(
                            f"\r  {dest.name}: {done / 1024**2:8.1f} / "
                            f"{total / 1024**2:.1f} MB ({pct:5.1f}%)"
                        )
                        sys.stderr.flush()
            if total and sys.stderr.isatty():
                sys.stderr.write("\n")
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        if exc.code in (401, 403):
            raise KaggleAuthError(
                "Kaggle rejected the credentials (HTTP "
                f"{exc.code}). Check the token is current, and that you have accepted any "
                "terms shown on the dataset page while signed in."
            ) from exc
        raise
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"network error downloading {dest.name}: {exc.reason}") from exc

    # Kaggle serves single files wrapped in a zip. Unwrap so data/raw holds the real artefact.
    if expect_zip and zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if len(names) == 1:
                with zf.open(names[0]) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                tmp.unlink()
                return dest
            zf.extractall(dest.parent)
        tmp.unlink()
        return dest.parent / names[0] if names else dest

    tmp.replace(dest)
    return dest


def download_lending_club(
    out_dir: str | Path,
    files: tuple[str, ...] = (ACCEPTED_FILE,),
    dataset: str = KAGGLE_DATASET,
    force: bool = False,
) -> dict[str, str]:
    """Download the named files into ``out_dir`` and record their SHA-256 checksums.

    Re-running is cheap: a file whose checksum already matches the recorded value is skipped.
    Pinning checksums matters here because the source is a third-party mirror of data the
    original publisher no longer hosts — results should stay reproducible against a fixed copy.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checksum_path = out_dir / CHECKSUM_FILE
    recorded: dict[str, str] = {}
    if checksum_path.exists():
        recorded = json.loads(checksum_path.read_text(encoding="utf-8"))

    credentials = load_credentials()
    LOG.info("authenticated to Kaggle as %s", credentials.username)

    for name in files:
        dest = out_dir / name
        if dest.exists() and not force:
            if recorded.get(name) and recorded[name] == sha256sum(dest):
                LOG.info("%s already present and checksum matches; skipping", name)
                continue
            LOG.warning("%s present but checksum missing or stale; re-downloading", name)

        url = (
            f"{KAGGLE_API_ROOT}/datasets/download/{dataset}/"
            f"{urllib.parse.quote(name)}"
        )
        LOG.info("downloading %s", name)
        _download(url, dest, credentials, expect_zip=True)
        recorded[name] = sha256sum(dest)
        LOG.info("%s -> %.1f MB, sha256 %s", name, dest.stat().st_size / 1024**2, recorded[name][:16])

    checksum_path.write_text(json.dumps(recorded, indent=2, sort_keys=True), encoding="utf-8")
    marker = out_dir / "SYNTHETIC"
    if marker.exists():
        marker.unlink()
        LOG.info("removed SYNTHETIC marker: this directory now holds real data")
    return recorded


def verify_raw(out_dir: str | Path, files: tuple[str, ...] = (ACCEPTED_FILE,)) -> bool:
    """Return True if every named file is present and matches its recorded checksum."""
    out_dir = Path(out_dir)
    checksum_path = out_dir / CHECKSUM_FILE
    if not checksum_path.exists():
        return False
    recorded = json.loads(checksum_path.read_text(encoding="utf-8"))
    for name in files:
        path = out_dir / name
        if not path.exists() or recorded.get(name) != sha256sum(path):
            return False
    return True


def detect_data_source(raw_dir: str | Path) -> str:
    """Return ``"synthetic"``, ``"real"`` or ``"absent"`` for the raw directory.

    Every manifest records this. It is the mechanism that stops a number produced from the
    synthetic generator being quoted as a result about consumer credit.
    """
    raw_dir = Path(raw_dir)
    if (raw_dir / "SYNTHETIC").exists():
        return "synthetic"
    if any(raw_dir.glob("*.csv.gz")) or any(raw_dir.glob("*.csv")):
        return "real"
    return "absent"

# params_loader.py
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable
from urllib.error import HTTPError, URLError

from flax.training import checkpoints


# ============================================================
# Helpers: checkpoint discovery
# ============================================================

def _find_ckpt_root(root: str, max_depth: int = 10) -> Optional[str]:
    """
    Return a directory that *contains* checkpoint_* directories (e.g. checkpoint_0),
    searching recursively. The returned path is the parent directory of checkpoint_*.

    Works with Orbax/OCDBT layouts like:
      <ckpt_root>/checkpoint_0/_METADATA ...
    """
    root_p = Path(root)
    if not root_p.exists() or not root_p.is_dir():
        return None

    # Direct: root contains checkpoint_*
    if any(root_p.glob("checkpoint_*")):
        return str(root_p)

    base_depth = len(root_p.resolve().parts)

    for cur, dirs, _files in os.walk(root_p):
        cur_p = Path(cur)
        depth = len(cur_p.resolve().parts) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue

        if any(d.startswith("checkpoint_") for d in dirs):
            return str(cur_p)

    return None


# ============================================================
# Helpers: Zenodo URL / API
# ============================================================

def _parse_zenodo_recid(record_url: str) -> Optional[str]:
    """
    Accepts:
      https://zenodo.org/records/<RECID>
      https://zenodo.org/record/<RECID>
      https://zenodo.org/records/<RECID>/...
    """
    m = re.search(r"zenodo\.org/(?:records|record)/(\d+)", record_url)
    return m.group(1) if m else None


def _zenodo_api_record(recid: str) -> dict:
    api_url = f"https://zenodo.org/api/records/{recid}"
    with urllib.request.urlopen(api_url) as r:
        return json.load(r)


def _zenodo_pick_download_link(record_url: str, asset_name: Optional[str]) -> tuple[str, str]:
    """
    Returns (download_url, resolved_asset_name).
    Uses Zenodo API so we don't rely on brittle /files/... manual URLs.
    """
    recid = _parse_zenodo_recid(record_url)
    if recid is None:
        raise ValueError(
            f"Could not parse Zenodo record id from: {record_url}\n"
            "Expected something like https://zenodo.org/records/<RECID>"
        )

    payload = _zenodo_api_record(recid)
    files = payload.get("files", []) or []

    if not files:
        raise FileNotFoundError(f"Zenodo record {recid} has no files.")

    # Helper getters
    def key(f: dict) -> str:
        return f.get("key") or ""

    def dl(f: dict) -> Optional[str]:
        links = f.get("links", {}) or {}
        return links.get("download") or links.get("self")

    # If asset_name provided, match exactly
    if asset_name is not None:
        for f in files:
            if key(f) == asset_name:
                url = dl(f)
                if not url:
                    raise FileNotFoundError(f"Zenodo file found but has no download link: {asset_name}")
                return url, asset_name
        available = [key(f) for f in files]
        raise FileNotFoundError(
            f"Zenodo asset '{asset_name}' not found in record {recid}.\n"
            f"Available: {available}"
        )

    # Otherwise: prefer a single zip; if multiple zips, pick the first
    zip_files = [f for f in files if key(f).endswith(".zip")]
    if zip_files:
        url = dl(zip_files[0])
        if not url:
            raise FileNotFoundError("Found zip on Zenodo but no download link.")
        return url, key(zip_files[0])

    # Fallback: first file
    url = dl(files[0])
    if not url:
        raise FileNotFoundError("Found files on Zenodo but no download link.")
    return url, key(files[0])


# ============================================================
# Helpers: download + extract
# ============================================================

def _download_file(url: str, dst_path: str) -> None:
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst_path, "wb") as f:
        shutil.copyfileobj(r, f)


def _safe_extract_zip(zip_path: str, dst_dir: str) -> None:
    """
    Safe zip extraction (prevents Zip Slip path traversal).
    """
    dst = Path(dst_dir).resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            out_path = (dst / member).resolve()
            if not str(out_path).startswith(str(dst) + os.sep) and out_path != dst:
                raise RuntimeError(f"Unsafe path in zip: {member}")
        zf.extractall(dst_dir)


def _extract_archive(archive_path: str, dst_dir: str) -> None:
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    ap = str(archive_path)

    if ap.endswith(".zip"):
        _safe_extract_zip(ap, dst_dir)
        return

    if ap.endswith(".tar") or ap.endswith(".tar.gz") or ap.endswith(".tgz"):
        mode = "r:gz" if (ap.endswith(".tar.gz") or ap.endswith(".tgz")) else "r:"
        with tarfile.open(ap, mode) as tf:
            tf.extractall(dst_dir)
        return

    raise ValueError(f"Unsupported archive format: {archive_path}")


def _archive_top_level_dirs(archive_path: str) -> set[str]:
    """
    Inspect archive to detect top-level folder names (e.g. 'checkpoints_new').
    Used to choose a sane extraction destination automatically.
    """
    ap = str(archive_path)
    top = set()

    if ap.endswith(".zip"):
        with zipfile.ZipFile(ap, "r") as zf:
            for n in zf.namelist():
                n = n.lstrip("/")
                if not n:
                    continue
                top.add(n.split("/", 1)[0])
        return top

    if ap.endswith(".tar") or ap.endswith(".tar.gz") or ap.endswith(".tgz"):
        mode = "r:gz" if (ap.endswith(".tar.gz") or ap.endswith(".tgz")) else "r:"
        with tarfile.open(ap, mode) as tf:
            for m in tf.getmembers():
                n = (m.name or "").lstrip("/")
                if not n:
                    continue
                top.add(n.split("/", 1)[0])
        return top

    return top


def _default_extract_dir_for_ckpt(ckpt_dir: str, archive_path: str) -> str:
    """
    Choose an extraction directory that avoids creating nested 'checkpoints_new/checkpoints_new/...'
    when ckpt_dir points inside the bundle.

    Heuristic:
    - If archive contains a top-level folder named 'checkpoints_new' AND ckpt_dir contains '/checkpoints_new/',
      then extract into the parent of that 'checkpoints_new' folder.
    - Otherwise extract into parent(ckpt_dir).
    """
    ckpt_p = Path(ckpt_dir).resolve()
    top = _archive_top_level_dirs(archive_path)

    # Common case: bundle contains "checkpoints_new/..."
    if "checkpoints_new" in top:
        parts = ckpt_p.parts
        if "checkpoints_new" in parts:
            idx = parts.index("checkpoints_new")
            # extract to the directory *above* checkpoints_new
            return str(Path(*parts[:idx]).resolve())

    return str(ckpt_p.parent.resolve())


# ============================================================
# Public API
# ============================================================

@dataclass(frozen=True)
class ZenodoSource:
    record_url: str                     # e.g. "https://zenodo.org/records/19736519"
    asset_name: Optional[str] = None    # e.g. "checkpoints_new.zip"


def restore_checkpoint_or_zenodo(
    ckpt_dir: str,
    target,
    zenodo: ZenodoSource,
    *,
    cache_dir: Optional[str] = None,
    force_redownload: bool = False,
    verbose: bool = True,
):
    """
    Restore a Flax/Orbax checkpoint from ckpt_dir.
    If missing or restore fails, download+extract an archive from Zenodo and retry.

    Parameters
    ----------
    ckpt_dir:
      The *local* directory where the checkpoint should exist, e.g.
        ".../checkpoints_new/deeponet_params_new_log15_time_diff"
      This is the directory that (after extraction) should contain "checkpoint_0/..."

    target:
      The target PyTree / TrainState skeleton.

    zenodo:
      ZenodoSource(record_url="https://zenodo.org/records/<RECID>",
                   asset_name="checkpoints_new.zip")

    cache_dir:
      Where to store the downloaded archive before extraction.
      Default: "<ckpt_dir>/../.zenodo_cache" (one level above).

    Notes
    -----
    - Supports bundle archives (e.g. checkpoints_new.zip containing multiple model folders).
    - Supports single-model archives (containing checkpoint_0 directly).
    """
    ckpt_dir = os.path.abspath(ckpt_dir)

    # 1) Try local restore first
    local_root = _find_ckpt_root(ckpt_dir)
    if local_root is not None:
        try:
            if verbose:
                print(f"[ckpt] Restoring locally from: {local_root}")
            return checkpoints.restore_checkpoint(local_root, target=target)
        except Exception as e:
            if verbose:
                print(f"[ckpt] Local restore failed ({type(e).__name__}: {e}). Will try Zenodo...")

    # 2) Prepare cache
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(ckpt_dir), ".zenodo_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 3) Resolve download link via Zenodo API
    download_url, resolved_asset_name = _zenodo_pick_download_link(
        zenodo.record_url, zenodo.asset_name
    )

    archive_path = os.path.join(cache_dir, resolved_asset_name)

    # 4) Download if needed
    need_download = force_redownload or (not os.path.exists(archive_path))
    if need_download:
        if verbose:
            print(f"[ckpt] Downloading Zenodo asset '{resolved_asset_name}'")
            print(f"[ckpt]  from: {download_url}")
            print(f"[ckpt]  to:   {archive_path}")
        try:
            _download_file(download_url, archive_path)
        except (HTTPError, URLError) as e:
            raise RuntimeError(
                f"Failed to download Zenodo asset '{resolved_asset_name}' from {download_url}\n"
                f"Error: {type(e).__name__}: {e}"
            ) from e
    else:
        if verbose:
            print(f"[ckpt] Using cached Zenodo archive: {archive_path}")

    # 5) Extract to a sensible destination (avoid nested folders)
    extract_dir = _default_extract_dir_for_ckpt(ckpt_dir, archive_path)
    if verbose:
        print(f"[ckpt] Extracting archive into: {extract_dir}")
    _extract_archive(archive_path, extract_dir)

    # 6) Retry restore
    extracted_root = _find_ckpt_root(ckpt_dir)
    if extracted_root is None:
        # last resort: maybe ckpt_dir points too deep; try to find something near extract_dir
        fallback = _find_ckpt_root(extract_dir)
        raise FileNotFoundError(
            f"Checkpoint not found after extraction.\n"
            f"Expected under ckpt_dir: {ckpt_dir}\n"
            f"Found checkpoint root under extract_dir: {fallback}\n"
            f"Hint: check that ckpt_dir matches the folder structure inside the archive."
        )

    if verbose:
        print(f"[ckpt] Restoring after extraction from: {extracted_root}")
    return checkpoints.restore_checkpoint(extracted_root, target=target)

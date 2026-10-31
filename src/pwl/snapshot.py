"""Snapshot persistence with integrity hashes."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DATA_EXT
from .logging_config import get_logger

logger = get_logger(__name__)


def save_snapshot(df: pd.DataFrame, name: str, source: str) -> Path:
    """Write a dataframe to data/external and record its hash in manifest.json."""
    DATA_EXT.mkdir(parents=True, exist_ok=True)
    path = DATA_EXT / f"{name}.csv"
    df.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest_path = DATA_EXT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[name] = {
        "sha256": digest,
        "rows": int(len(df)),
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    logger.info("Saved snapshot %s: %d rows, sha256=%s", name, len(df), digest[:16])
    return path


def verify_snapshot(name: str) -> bool:
    """Verify a snapshot against its manifest hash."""
    manifest_path = DATA_EXT / "manifest.json"
    if not manifest_path.exists():
        logger.warning("No manifest found for verification")
        return False

    manifest = json.loads(manifest_path.read_text())
    if name not in manifest:
        logger.warning("Snapshot %s not in manifest", name)
        return False

    path = DATA_EXT / f"{name}.csv"
    if not path.exists():
        logger.warning("Snapshot file %s missing", path)
        return False

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = manifest[name]["sha256"]
    if digest != expected:
        logger.error("Hash mismatch for %s: expected %s, got %s", name, expected[:16], digest[:16])
        return False

    logger.info("Snapshot %s verified", name)
    return True

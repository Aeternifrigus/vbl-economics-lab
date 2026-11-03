"""Versioned, verifiable model releases.

A release is a folder (and a tar.gz of it) holding the model, its metadata, the exact data it was
trained on, the evaluation and the model card, with a manifest of SHA-256 hashes. `verify_release`
re-checks every hash, so a downloaded release can be trusted or rejected without the repository.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .exceptions import ReleaseError
from .logging_config import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "RELEASE_MANIFEST.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def latest_model_files(models_dir: Path) -> tuple[Path, Path]:
    """Return (joblib, metadata json) of the newest trained lag model."""
    candidates = sorted(models_dir.glob("lagged_gm_*.json"))
    if not candidates:
        raise ReleaseError("no trained model found. Run `pwl train` first.")
    meta = candidates[-1]
    model = meta.with_suffix(".joblib")
    if not model.exists():
        raise ReleaseError(f"{model.name} is missing next to its metadata")
    return model, meta


def build_release(
    version: str,
    models_dir: Path,
    features_csv: Path,
    vbl_csv: Path,
    reports_dir: Path,
    docs: list[Path],
    out_dir: Path,
    repo_root: Path,
) -> Path:
    """Assemble and archive a release. Refuses a model that was not trained on the current data."""
    model, meta_path = latest_model_files(models_dir)
    meta = json.loads(meta_path.read_text())
    trained_on = meta.get("data_sha256")
    current = file_sha256(features_csv)
    if trained_on != current:
        raise ReleaseError(
            "the model is stale: it was trained on different feature data than the current "
            "snapshot. Run `pwl features` and `pwl train` again.",
            problems=[f"model data_sha256 {str(trained_on)[:12]} != current {current[:12]}"],
        )

    name = f"vbl-economics-lab-{version}"
    bundle = out_dir / name
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "model").mkdir(parents=True)
    (bundle / "data").mkdir()
    (bundle / "reports").mkdir()

    copies = {
        f"model/{model.name}": model,
        f"model/{meta_path.name}": meta_path,
        "data/features_quarterly.csv": features_csv,
        "data/vbl_quarterly.csv": vbl_csv,
    }
    for report in ("backtest.json", "operating_leverage.md", "nowcast.json", "monitor.json"):
        if (reports_dir / report).exists():
            copies[f"reports/{report}"] = reports_dir / report
    for doc in docs:
        if doc.exists():
            copies[doc.name] = doc
    for rel, src in copies.items():
        shutil.copyfile(src, bundle / rel)

    backtest_path = reports_dir / "backtest.json"
    backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else {}
    manifest: dict[str, Any] = {
        "version": version,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(repo_root),
        "model_latest_quarter": meta.get("latest_quarter"),
        "oil_effect_gm_points_per_10usd": meta.get("gm_points_per_10usd_brent"),
        "backtest": {
            k: backtest.get(k) for k in ("n_test_quarters", "mae_model", "mae_seasonal_naive")
        },
        "files": {
            rel: {"sha256": file_sha256(bundle / rel), "bytes": (bundle / rel).stat().st_size}
            for rel in sorted(copies)
        },
    }
    (bundle / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    archive = Path(shutil.make_archive(str(bundle), "gztar", root_dir=out_dir, base_dir=name))
    Path(f"{archive}.sha256").write_text(f"{file_sha256(archive)}  {archive.name}\n")
    logger.info("Built release %s (%d files)", name, len(copies))
    return archive


def verify_release(bundle: Path) -> list[str]:
    """Re-check every file hash in a release folder. Returns problems (empty means intact)."""
    manifest_path = bundle / MANIFEST_NAME
    if not manifest_path.exists():
        return [f"{MANIFEST_NAME} is missing"]
    manifest = json.loads(manifest_path.read_text())
    problems = []
    for rel, info in manifest["files"].items():
        path = bundle / rel
        if not path.exists():
            problems.append(f"{rel}: missing")
        elif file_sha256(path) != info["sha256"]:
            problems.append(f"{rel}: hash mismatch")
    return problems

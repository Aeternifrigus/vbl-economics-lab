import hashlib
import json
from datetime import datetime, timezone

from .config import DATA_EXT


def save_snapshot(df, name, source):
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
    return path

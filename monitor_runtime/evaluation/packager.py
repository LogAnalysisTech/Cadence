from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import modal


VOLUME_NAME = "rafd-mini-swe-agent-artifacts"

app = modal.App("rafd-artifact-packager")
artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/artifacts": artifact_volume},
    timeout=60 * 60,
)
def package_run(run_id: str) -> dict[str, object]:
    source = Path("/artifacts") / run_id
    if not source.is_dir():
        raise FileNotFoundError(f"Artifact run does not exist: {source}")

    archive_dir = Path("/artifacts") / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{run_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=run_id)

    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    artifact_volume.commit()
    return {
        "run_id": run_id,
        "archive_path": f"/archives/{archive.name}",
        "size_bytes": archive.stat().st_size,
        "sha256": digest.hexdigest(),
    }


@app.local_entrypoint()
def main(run_id: str) -> None:
    print(json.dumps(package_run.remote(run_id), indent=2))

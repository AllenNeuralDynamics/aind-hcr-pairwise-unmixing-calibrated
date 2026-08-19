"""Asset manifest: what the registered data asset for this run should look like.

The capsule CANNOT register its own result. Code Ocean uploads /results to S3 only
AFTER the run script exits, so at the moment this code runs there is nothing to point a
data asset at. What the capsule can do -- and what this module does -- is write down
exactly what the asset should be called, described as, and tagged with, so a caller
outside the run can register it without re-deriving any of it by hand.

Consumed by tools/register_result_asset.py, which reads results/asset_manifest.json from
the finished computation and creates the asset via the Code Ocean API.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

#: Written into the asset name and data_description.process_name.
PROCESS_SLUG = "unmixed-calibrated"

#: Tags applied to the registered asset. These match the tags already carried by the
#: HCR assets in this project so the new asset turns up in the same searches.
DEFAULT_TAGS = ("HCR", "learning_mfish", "cell types and learning", PROCESS_SLUG)


def classify_mounts(mount_names, mouse_id):
    """Split mounted data-asset folder names by kind, for THIS mouse.

    Code Ocean mounts every attached asset as a folder under /root/capsule/data, so the
    directory listing IS the list of attached assets. Assets for other mice are reported
    separately rather than dropped: they contributed nothing to the output, and saying so
    in the description is more useful than silently omitting them.
    """
    mine = lambda n: re.match(rf"HCR_{mouse_id}(_|$)", n) is not None
    out = dict(unmixing=[], processed=[], raw=[], other_mouse=[])
    for n in sorted(mount_names):
        if not mine(n):
            out["other_mouse"].append(n)
        elif "pairwise-unmixing" in n:
            out["unmixing"].append(n)
        elif "_processed_" in n:
            out["processed"].append(n)
        else:
            out["raw"].append(n)
    return out


def _bullet(title, names):
    if not names:
        return f"{title}: none\n"
    return f"{title} ({len(names)}):\n" + "".join(f"  - {n}\n" for n in names)


def build_description(mouse_id, rounds, inputs, n_cells=None, n_genes=None,
                      capsule_name=None, extra=None):
    """Human-readable description naming every input asset the run consumed."""
    lines = [
        f"Spectrally unmixed spot tables and cell x gene table for mouse {mouse_id}, "
        f"rounds {', '.join(rounds)}.",
    ]
    if n_cells is not None and n_genes is not None:
        lines.append(f"Cell x gene table: {n_cells:,} cells x {n_genes} genes.")
    if capsule_name:
        lines.append(f"Produced by {capsule_name}.")
    lines.append("")
    lines.append("INPUT DATA ASSETS")
    lines.append(_bullet("Unmixing input (mixed spot tables)", inputs["unmixing"]).rstrip())
    lines.append(_bullet("Processed assets (acquisition.json, image_spot_detection fg/bg)",
                         inputs["processed"]).rstrip())
    lines.append(_bullet("Raw acquisition assets", inputs["raw"]).rstrip())
    if inputs["other_mouse"]:
        lines.append("")
        lines.append(_bullet("Also mounted but NOT used by this run (different mouse)",
                             inputs["other_mouse"]).rstrip())
    if extra:
        lines.append("")
        lines.append(extra)
    return "\n".join(lines)


def asset_name(mouse_id, creation_time=None, process_slug=PROCESS_SLUG):
    """HCR_<mouse>_<slug>_<YYYY-MM-DD>_<HH-MM-SS>, in UTC.

    Deliberately keyed on the MOUSE, not on one parent session: this capsule consumes
    every round of a mouse, so naming it after a single processed session would assert a
    parentage that is only one fifth true.
    """
    t = creation_time or datetime.now(timezone.utc)
    return f"HCR_{mouse_id}_{process_slug}_{t.strftime('%Y-%m-%d_%H-%M-%S')}"


def write_manifest(output_dir, mouse_id, rounds, data_dir="/root/capsule/data",
                   n_cells=None, n_genes=None, creation_time=None,
                   capsule_name=None, tags=DEFAULT_TAGS, experimenter=None,
                   extra_description=None):
    """Write results/asset_manifest.json. Returns the manifest dict."""
    t = creation_time or datetime.now(timezone.utc)
    data_p = Path(data_dir)
    mounts = sorted(p.name for p in data_p.iterdir() if p.is_dir()) if data_p.exists() else []
    inputs = classify_mounts(mounts, mouse_id)
    name = asset_name(mouse_id, t)
    manifest = {
        "name": name,
        "mount": name,
        "tags": list(tags),
        "description": build_description(mouse_id, rounds, inputs, n_cells, n_genes,
                                         capsule_name, extra_description),
        "custom_metadata": {
            "data level": "derived",
            "experiment type": "HCR",
            "institution": "AIND",
            "modality": "Selective plane illumination microscopy",
            "subject id": str(mouse_id),
            "subject species": "Mus musculus",
        },
        "input_assets": inputs,
        "rounds": list(rounds),
        "process_name": PROCESS_SLUG,
        "creation_time": t.isoformat().replace("+00:00", "Z"),
    }
    if experimenter:
        manifest["experimenter"] = experimenter
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    path = outp / "asset_manifest.json"
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest

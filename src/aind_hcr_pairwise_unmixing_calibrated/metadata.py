"""aind-data-schema metadata for the derived asset this capsule produces.

Two jobs:

  1. Carry the upstream schema files (subject.json, data_description.json,
     acquisition.json, procedures.json, instrument.json, quality_control.json ...)
     forward into the results directory, so the derived asset is not orphaned from
     the metadata describing where it came from.

  2. Append one DataProcess describing THIS step to processing.json, preserving any
     steps already recorded upstream.

SCHEMA VERSION. This module writes **1.1.4**, matching the processing.json files
already present in the HCR processed assets (e.g.
HCR_800995_..._processed_.../cell_body_segmentation/processing.json). The installed
aind-data-schema library is 2.x and its `Processing` model emits schema 2.3.0 with a
different structure -- a flat `data_processes` list, `process_type`/`stage`/`code`
fields, required `experimenters`. Mixing the two in one asset would leave a file that
neither reader handles cleanly, so the shape here is written by hand to match what is
already in the assets. Migrating to 2.x is a deliberate future step, not something to
do implicitly: see `_V114_DATA_PROCESS_FIELDS` and `upgrade_note`.

The process name "Image spot spectral unmixing" is a term in the aind-data-schema
controlled vocabulary (ProcessName), so it stays valid across the version change.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.1.4"
DESCRIBED_BY = ("https://raw.githubusercontent.com/AllenNeuralDynamics/aind-data-schema/"
                "main/src/aind_data_schema/core/processing.py")

#: Field order of a 1.1.4 DataProcess, as found in the existing HCR assets.
_V114_DATA_PROCESS_FIELDS = ("name", "software_version", "start_date_time",
                             "end_date_time", "input_location", "output_location",
                             "code_url", "code_version", "parameters", "outputs",
                             "notes", "resources")

#: Controlled-vocabulary process name (aind_data_schema_models.process_names).
PROCESS_NAME = "Image spot spectral unmixing"

CODE_URL = "https://github.com/AllenNeuralDynamics/aind-hcr-pairwise-unmixing-calibrated"

#: Core schema files carried into the derived asset unchanged. These describe the
#: SUBJECT and the ACQUISITION, which our processing does not alter, so copying them
#: forward is correct.
#:
#: data_description.json is deliberately NOT in this list -- see
#: derived_data_description(). It describes the ASSET, and a derived asset must name
#: itself and point at its parent; copying the parent's would claim our output is the
#: parent. metadata.nd.json is also excluded: it is an aggregate of the others and
#: would go stale the moment data_description differs.
CORE_METADATA_FILES = ("subject.json", "procedures.json", "instrument.json",
                       "rig.json", "session.json", "acquisition.json",
                       "quality_control.json")

#: Name of this processing step, used in the derived asset name and in
#: data_description.process_name.
PROCESS_SLUG = "unmixed-calibrated"


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def copy_upstream_metadata(source_dirs, output_dir, files=CORE_METADATA_FILES):
    """Copy core schema files from upstream asset(s) into the results directory.

    source_dirs  directories to search, in priority order; the FIRST hit for a given
                 filename wins, so put the most specific asset first.

    Returns {filename: source path} for what was copied.
    """
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    copied = {}
    for fname in files:
        for src_dir in source_dirs:
            if src_dir is None:
                continue
            src = Path(src_dir) / fname
            if src.exists() and fname not in copied:
                shutil.copy(src, outp / fname)
                copied[fname] = str(src)
                break
    return copied


def derived_data_description(parent_dir, output_dir, process_slug=PROCESS_SLUG,
                             creation_time=None, investigators=None):
    """Write a data_description.json for the DERIVED asset this capsule produces.

    A derived asset does not inherit its parent's data_description: that file names the
    asset, and copying it forward would assert that our output IS the parent. The AIND
    convention, read off the processed assets themselves, is

        name            = <parent name>_<process_slug>_<creation timestamp>
        data_level      = "derived"
        input_data_name = <parent name>
        process_name    = <process_slug>

    with subject_id, institution, investigators, funding, licence, modality and platform
    carried over from the parent because they are unchanged by processing.

    Returns the written path, or None when the parent has no data_description.json to
    inherit those fields from -- in that case there is nothing trustworthy to write and
    inventing values would be worse than omitting the file.
    """
    parent_dd = Path(parent_dir) / "data_description.json"
    if not parent_dd.exists():
        return None
    with open(parent_dd) as fh:
        parent = json.load(fh)

    now = creation_time or datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    parent_name = parent.get("name") or Path(parent_dir).name

    doc = dict(parent)                      # inherit unchanged descriptive fields
    doc.update({
        "name": f"{parent_name}_{process_slug}_{stamp}",
        "data_level": "derived",
        "input_data_name": parent_name,
        "process_name": process_slug,
        "creation_time": now.isoformat().replace("+00:00", "Z"),
    })
    if investigators:
        doc["investigators"] = [{"name": n} for n in investigators]

    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)
    path = outp / "data_description.json"
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=3)
    return str(path)


def unmixing_data_process(input_locations, output_location, parameters,
                          outputs=None, notes="", software_version=None,
                          start_date_time=None, end_date_time=None):
    """One 1.1.4 DataProcess describing this unmixing run."""
    from . import __version__
    return {
        "name": PROCESS_NAME,
        "software_version": software_version or __version__,
        "start_date_time": start_date_time or _now(),
        "end_date_time": end_date_time or _now(),
        "input_location": list(input_locations),
        "output_location": str(output_location),
        "code_url": CODE_URL,
        "code_version": software_version or __version__,
        "parameters": parameters,
        "outputs": outputs or {},
        "notes": notes,
        "resources": None,
    }


def write_processing(output_dir, data_process, upstream_processing=None,
                     processor_full_name=None, pipeline_note=None):
    """Write processing.json, appending `data_process` to any upstream history.

    upstream_processing  path to an existing processing.json to extend. When it is
                         1.1.4-shaped its data_processes are preserved and this step
                         is appended. When it is 2.x-shaped (a flat `data_processes`
                         key) it is NOT rewritten -- the prior history is referenced
                         in `note` instead of being silently downgraded.
    """
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    prior, note_extra = [], ""
    if upstream_processing and Path(upstream_processing).exists():
        with open(upstream_processing) as fh:
            up = json.load(fh)
        up_ver = str(up.get("schema_version", ""))
        if "processing_pipeline" in up:
            prior = up["processing_pipeline"].get("data_processes", [])
        elif "data_processes" in up:
            # 2.x upstream: a different shape. Do not downgrade it into 1.1.4 --
            # that would lose fields (process_type, stage, code, experimenters).
            note_extra = (f" Upstream processing.json is schema {up_ver} (2.x shape); "
                          f"its {len(up['data_processes'])} step(s) are not merged here. "
                          f"Original: {upstream_processing}")

    doc = {
        "describedBy": DESCRIBED_BY,
        "schema_version": SCHEMA_VERSION,
        "processing_pipeline": {
            "data_processes": list(prior) + [data_process],
            "processor_full_name": processor_full_name or "",
            "pipeline_version": data_process.get("code_version", ""),
            "pipeline_url": CODE_URL,
            "note": ((pipeline_note or "Calibrated pairwise crosstalk removal for "
                      "thick-tissue HCR spot tables.") + note_extra).strip(),
        },
        "analyses": [],
        "notes": "",
    }
    path = outp / "processing.json"
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=3)
    return str(path)


def find_upstream_processing(search_dirs):
    """First processing.json found under any of `search_dirs` (recursively)."""
    for d in search_dirs:
        if d is None:
            continue
        root = Path(d)
        if not root.is_dir():
            continue
        direct = root / "processing.json"
        if direct.exists():
            return str(direct)
        hits = sorted(root.rglob("processing.json"))
        if hits:
            return str(hits[0])
    return None


def upgrade_note():
    """Why this writes 1.1.4 and what a 2.x migration would involve."""
    return (
        "Written as schema 1.1.4 to match the processing.json already present in the "
        "HCR processed assets. To migrate to 2.x: build "
        "aind_data_schema.core.processing.Processing with a flat data_processes list of "
        "DataProcess(process_type=ProcessName.IMAGE_SPOT_SPECTRAL_UNMIXING, "
        "stage=ProcessStage.PROCESSING, code=Code(url=...), experimenters=[...]) and call "
        "write_standard_file(). Verified to validate under aind-data-schema 2.8.1. Do it "
        "for the whole asset at once, not per step."
    )

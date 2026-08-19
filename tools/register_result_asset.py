#!/usr/bin/env python
"""Register capsule results as Code Ocean data assets.

A capsule cannot register its own result: Code Ocean uploads /results to S3 only AFTER
the run script exits, so while the run is executing there is nothing to point an asset
at. This script closes that gap from outside the run, using the asset_manifest.json the
run writes into /results.

MANUAL -- register one run
    register_result_asset.py <computation_id>
    register_result_asset.py --latest                  # newest run that is ready
    register_result_asset.py --list                     # show candidates, register nothing

AUTOMATIC -- sweep a capsule (cron, scheduled capsule, CI)
    register_result_asset.py --watch
    register_result_asset.py --watch --interval 600      # keep polling

Environment:
    CODEOCEAN_DOMAIN      e.g. https://codeocean.allenneuraldynamics.org
    CODEOCEAN_TOKEN       needs data-asset create scope; never hard-code it
    CODEOCEAN_CAPSULE_ID  default for --capsule

Every mode is idempotent. The manifest name embeds the run's own UTC timestamp, so it is
unique per run and doubles as the key: a run whose asset already exists is skipped rather
than duplicated. --dry-run shows the plan without creating anything.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

TIMEOUT = 60
MANIFEST_NAME = "asset_manifest.json"


# --------------------------------------------------------------------------- API


def _api(path, token, domain, method="GET", body=None, soft=False):
    """Call the Code Ocean v1 API. soft=True returns None on 4xx instead of exiting."""
    url = f"{domain.rstrip('/')}/api/v1/{path.lstrip('/')}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    tok = base64.b64encode(f"{token}:".encode()).decode()
    req.add_header("Authorization", f"Basic {tok}")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            txt = resp.read().decode() or "{}"
    except urllib.error.HTTPError as exc:
        if soft and 400 <= exc.code < 500:
            return None
        raise SystemExit(f"API {method} {path} failed: {exc.code} "
                         f"{exc.read().decode()[:400]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"API {method} {path} unreachable: {exc}")
    return _loads_multi(txt)


def _loads_multi(txt):
    """Parse the response body.

    Some Code Ocean endpoints return several JSON objects concatenated rather than a
    JSON array (observed on capsules/<id>/computations), which json.loads rejects.
    """
    dec, i, out = json.JSONDecoder(), 0, []
    while i < len(txt):
        while i < len(txt) and txt[i] in " \n\t\r":
            i += 1
        if i >= len(txt):
            break
        obj, i = dec.raw_decode(txt, i)
        out.append(obj)
    if not out:
        return {}
    return out[0] if len(out) == 1 else out


def list_computations(capsule_id, token, domain):
    r = _api(f"capsules/{capsule_id}/computations", token, domain)
    if isinstance(r, dict):
        r = r.get("result") or r.get("items") or [r]
    return [c for c in (r or []) if isinstance(c, dict)]


def read_manifest(computation_id, token, domain):
    """results/asset_manifest.json from a finished computation, or None."""
    urls = _api(f"computations/{computation_id}/results/download_url", token, domain,
                method="POST", body={"path": MANIFEST_NAME}, soft=True)
    url = (urls or {}).get("url") or (urls or {}).get("download_url")
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return None


def asset_exists(name, token, domain):
    r = _api("data_assets/search", token, domain, method="POST",
             body={"query": f'name:"{name}"', "limit": 50}, soft=True)
    items = (r or {}).get("results") or (r or {}).get("items") or []
    # Compact search results use "n" for name; full records use "name".
    return any((it.get("name") or it.get("n")) == name for it in items)


def create_asset(computation_id, manifest, token, domain):
    body = {
        "name": manifest["name"],
        "mount": manifest.get("mount", manifest["name"]),
        "tags": manifest.get("tags", []),
        "description": manifest.get("description", ""),
        "source": {"computation": {"id": computation_id}},
    }
    if manifest.get("custom_metadata"):
        body["custom_metadata"] = manifest["custom_metadata"]
    return _api("data_assets", token, domain, method="POST", body=body)


# ----------------------------------------------------------------------- gating


SKIP_FAILED = "run failed (exit_code={code}) -- a failed run must not become an asset"
SKIP_NO_RESULTS = "no results to capture"
SKIP_NOT_DONE = "state={state}, not finished"
SKIP_NO_MANIFEST = (f"no {MANIFEST_NAME} in results -- run predates this feature, "
                    "or used --no-metadata")


def gate(comp):
    """Cheap checks from the computation record alone. Returns None when it may proceed."""
    if comp.get("state") not in (None, "completed"):
        return SKIP_NOT_DONE.format(state=comp.get("state"))
    if comp.get("exit_code") not in (0, None, "0"):
        return SKIP_FAILED.format(code=comp.get("exit_code"))
    if comp.get("has_results") is False:
        return SKIP_NO_RESULTS
    return None


def inspect(comp, token, domain):
    """(status, detail, manifest). status in {ready, registered, skip}."""
    reason = gate(comp)
    if reason:
        return "skip", reason, None
    man = read_manifest(comp["id"], token, domain)
    if not man:
        return "skip", SKIP_NO_MANIFEST, None
    if asset_exists(man["name"], token, domain):
        return "registered", man["name"], man
    return "ready", man["name"], man


def _when(comp):
    ts = comp.get("created")
    if not ts:
        return "?"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M")


# -------------------------------------------------------------------- commands


def cmd_list(comps, token, domain):
    print(f"{'computation':10s}  {'run (UTC)':16s}  {'status':11s} detail")
    n_ready = 0
    for c in sorted(comps, key=lambda z: z.get("created", 0), reverse=True):
        status, detail, _ = inspect(c, token, domain)
        n_ready += status == "ready"
        print(f"{c['id'][:8]:10s}  {_when(c):16s}  {status:11s} {detail}")
    print(f"\n{n_ready} run(s) ready to register.")
    return 0


def cmd_register(comps, token, domain, dry_run=False, only_latest=False):
    ordered = sorted(comps, key=lambda z: z.get("created", 0), reverse=True)
    todo = []
    for c in ordered:
        status, detail, man = inspect(c, token, domain)
        if status == "ready":
            todo.append((c, man))
            if only_latest:
                break
        elif not only_latest:
            print(f"{c['id'][:8]}  skipped      {detail}")
        elif status == "registered":
            # --latest: the newest run is already done; say so rather than walking back
            # to an older one the user did not ask about.
            print(f"{c['id'][:8]}  already registered as {detail}")
            return 0
    if not todo:
        print("nothing to register.")
        return 0
    for c, man in todo:
        print(f"\n{c['id']}  ({_when(c)} UTC)")
        print(f"  name : {man['name']}")
        print(f"  tags : {', '.join(man.get('tags', []))}")
        first = (man.get("description") or "").split("\n")[0]
        print(f"  desc : {first}")
        inputs = man.get("input_assets") or {}
        if inputs:
            print("  input assets: " + ", ".join(
                f"{k}={len(v)}" for k, v in inputs.items() if v))
        if dry_run:
            print("  --dry-run: not created")
            continue
        asset = create_asset(c["id"], man, token, domain)
        print(f"  created data asset {asset.get('id')}  state={asset.get('state')}")
    return 0


# ------------------------------------------------------------------------ main


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Register capsule results as Code Ocean data assets.",
        epilog="Manual: pass a computation_id, or --latest. Automatic: --watch from cron.")
    ap.add_argument("computation_id", nargs="?",
                    help="register this one run (manual mode)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--latest", action="store_true",
                      help="register the newest run that is ready (manual mode)")
    mode.add_argument("--list", action="store_true",
                      help="show every run and whether it is ready; register nothing")
    mode.add_argument("--watch", action="store_true",
                      help="register every ready run (for cron / scheduled use)")
    ap.add_argument("--capsule", default=os.environ.get("CODEOCEAN_CAPSULE_ID"),
                    help="capsule id; defaults to $CODEOCEAN_CAPSULE_ID")
    ap.add_argument("--interval", type=int, default=0, metavar="SEC",
                    help="with --watch, keep polling every SEC seconds")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be created, create nothing")
    args = ap.parse_args(argv)

    domain = os.environ.get("CODEOCEAN_DOMAIN")
    token = os.environ.get("CODEOCEAN_TOKEN")
    if not domain or not token:
        raise SystemExit("set CODEOCEAN_DOMAIN and CODEOCEAN_TOKEN")

    needs_capsule = args.latest or args.list or args.watch
    if not args.computation_id and not needs_capsule:
        raise SystemExit("give a computation_id, or one of --latest / --list / --watch")
    if needs_capsule and not args.capsule:
        raise SystemExit("--latest/--list/--watch need --capsule or $CODEOCEAN_CAPSULE_ID")

    def fetch():
        if args.computation_id:
            return [{"id": args.computation_id}]
        return list_computations(args.capsule, token, domain)

    if args.list:
        return cmd_list(fetch(), token, domain)

    rc = cmd_register(fetch(), token, domain, dry_run=args.dry_run,
                      only_latest=args.latest)
    while args.watch and args.interval:
        time.sleep(args.interval)
        rc = cmd_register(fetch(), token, domain, dry_run=args.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())

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
import urllib.parse
import urllib.request
from datetime import datetime, timezone

TIMEOUT = 60
MANIFEST_NAME = "asset_manifest.json"

#: Environment variables checked in order for each credential. A token attached to the
#: capsule as a Code Ocean SECRET arrives under the name declared in
#: .codeocean/secrets.json -- this capsule declares an api-key secret whose fields are
#: API_KEY and API_SECRET -- not under CODEOCEAN_TOKEN. Accepting both means an attached
#: secret works with no export, while an explicit export still wins if you set one.
#: CO_CAPSULE_ID is set by Code Ocean inside a capsule, so --capsule is usually redundant
#: there. Order matters: the explicit CODEOCEAN_* names are checked first so that an
#: export can always override an attached secret.
TOKEN_VARS = ("CODEOCEAN_TOKEN", "API_SECRET", "API_KEY", "CO_TOKEN")
DOMAIN_VARS = ("CODEOCEAN_DOMAIN", "CO_DOMAIN")
CAPSULE_VARS = ("CODEOCEAN_CAPSULE_ID", "CO_CAPSULE_ID")


def _first_env(names):
    """(value, name_it_came_from) for the first of `names` that is set and non-empty."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value, name
    return None, None


def _missing_credentials(domain, token):
    """Name what is missing, where it can come from, and how to check -- without ever
    printing a secret value."""
    lines = [""]
    if not token:
        lines += [
            "No API token found. Checked, in order: "
            + ", ".join("$" + v for v in TOKEN_VARS) + ".",
            "",
            "If the token is attached to the capsule as a secret, it appears under the",
            "name declared in .codeocean/secrets.json (this capsule declares API_KEY and",
            "API_SECRET). List the NAMES present without printing any value:",
            "",
            "    env | cut -d= -f1 | sort | grep -iE 'api|token|^co_'",
            "",
            "then export the one holding the token, e.g.:",
            "",
            "    export CODEOCEAN_TOKEN=\"$API_SECRET\"",
            "",
            "Secrets are only present in environments Code Ocean injects them into; a",
            "plain terminal in a cloud workstation may not have them.",
            "",
        ]
    if not domain:
        lines += [
            "No domain found. Checked: " + ", ".join("$" + v for v in DOMAIN_VARS) + ".",
            "",
            "    export CODEOCEAN_DOMAIN=https://codeocean.allenneuraldynamics.org",
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- API


def _api(path, token, domain, method="GET", body=None, params=None, soft=False):
    """Call the Code Ocean v1 API. soft=True returns None on 4xx instead of exiting."""
    url = f"{domain.rstrip('/')}/api/v1/{path.lstrip('/')}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
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


def get_computation(computation_id, token, domain):
    """Fetch one computation record.

    Needed because gate() reads state/end_status/exit_code/has_results off the record: an
    id alone carries none of those, and a dict holding only {"id": ...} passes every check
    vacuously. Naming a computation by hand must not be a way around the safety gate.
    """
    r = _api(f"computations/{computation_id}", token, domain)
    if isinstance(r, list):
        r = r[0] if r else {}
    if not isinstance(r, dict) or not r.get("id"):
        raise SystemExit(f"computation {computation_id} not found")
    return r


def read_manifest(computation_id, token, domain):
    """results/asset_manifest.json from a finished computation, or None.

    GET computations/<id>/results/urls?path=... , which is what the official client
    (codeocean 0.16.0, Computations.get_result_file_urls) calls. The older
    results/download_url route is deprecated there.
    """
    urls = _api(f"computations/{computation_id}/results/urls", token, domain,
                params={"path": MANIFEST_NAME}, soft=True)
    url = (urls or {}).get("download_url") or (urls or {}).get("url")
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return None


def asset_exists(name, token, domain):
    """True when an asset with exactly this name is already registered.

    Deliberately NOT soft: this is the only thing preventing duplicate assets, so a
    transient search failure must abort the run rather than be read as "no duplicate
    found". Failing open here would let a cron sweep create one extra asset per pass for
    as long as the search endpoint misbehaves.
    """
    r = _api("data_assets/search", token, domain, method="POST",
             body={"query": f'name:"{name}"', "limit": 50})
    if not isinstance(r, dict):
        raise SystemExit(f"unexpected data_assets/search response for {name!r}: "
                         f"{type(r).__name__} -- refusing to create a possible duplicate")
    items = r.get("results") or r.get("items") or []
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


SKIP_FAILED = "run did not succeed ({why}) -- a failed run must not become an asset"
SKIP_NO_RESULTS = "no results to capture"
SKIP_NOT_DONE = "state={state}, not finished"
SKIP_NO_MANIFEST = (f"no {MANIFEST_NAME} in results -- run predates this feature, "
                    "or used --no-metadata")


def gate(comp):
    """Cheap checks from the computation record alone. Returns None when it may proceed.

    end_status and exit_code are both checked because they mean different things and
    disagree in practice. On this capsule, f8ca3896 carries exit_code=0 with
    end_status="failed" (stopped before the script's own exit status meant anything),
    while d20036dc carries end_status="succeeded" with exit_code=1 (the script ran to
    completion and reported failure).

    Neither of those two would have slipped through without the end_status check -- both
    also have has_results=False, so the results check below already rejects them. The
    case end_status actually guards is a run stopped part-way that DID leave results:
    exit_code=0, has_results=True, end_status="failed". Not observed on this capsule,
    which is precisely why it is worth a cheap check rather than an assumption.
    """
    if comp.get("state") not in (None, "completed"):
        return SKIP_NOT_DONE.format(state=comp.get("state"))
    end_status = comp.get("end_status")
    if end_status is not None and end_status != "succeeded":
        return SKIP_FAILED.format(why=f"end_status={end_status}")
    if comp.get("exit_code") not in (0, None, "0"):
        return SKIP_FAILED.format(why=f"exit_code={comp.get('exit_code')}")
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
    """Register ready runs. only_latest stops at the newest ready one.

    --latest means "the newest run that can be registered", which is not always the newest
    run. When runs are passed over on the way there, each is named with its reason: silently
    registering an older run while the newest one failed would read as success for the run
    the user just did.
    """
    ordered = sorted(comps, key=lambda z: z.get("created", 0), reverse=True)
    todo = []
    for c in ordered:
        status, detail, man = inspect(c, token, domain)
        if status == "ready":
            todo.append((c, man))
            if only_latest:
                break
        elif status == "registered" and only_latest:
            # Stop rather than walking back: the newest run is accounted for, and an older
            # one is not what --latest was asked to do.
            print(f"{c['id'][:8]}  already registered as {detail}")
            return 0
        else:
            print(f"{c['id'][:8]}  skipped      {detail}")
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
    ap.add_argument("--capsule", default=_first_env(CAPSULE_VARS)[0],
                    help="capsule id; defaults to $CODEOCEAN_CAPSULE_ID or $CO_CAPSULE_ID")
    ap.add_argument("--interval", type=int, default=0, metavar="SEC",
                    help="with --watch, keep polling every SEC seconds")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be created, create nothing")
    args = ap.parse_args(argv)

    domain, domain_src = _first_env(DOMAIN_VARS)
    token, token_src = _first_env(TOKEN_VARS)
    if not domain or not token:
        raise SystemExit(_missing_credentials(domain, token))
    print(f"domain from ${domain_src}; token from ${token_src}", flush=True)

    needs_capsule = args.latest or args.list or args.watch
    if not args.computation_id and not needs_capsule:
        raise SystemExit("give a computation_id, or one of --latest / --list / --watch")
    if needs_capsule and not args.capsule:
        raise SystemExit(
            "--latest/--list/--watch need a capsule id: pass --capsule, or set "
            + " or ".join("$" + v for v in CAPSULE_VARS)
            + ". Inside a Code Ocean capsule $CO_CAPSULE_ID is set for you.")

    def fetch():
        if args.computation_id:
            # The full record, not just the id: gate() has nothing to check otherwise.
            return [get_computation(args.computation_id, token, domain)]
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

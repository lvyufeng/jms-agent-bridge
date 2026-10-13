#!/usr/bin/env python3
"""Helpers for ``scripts/verify-onboarding.sh``.

The acceptance script drives the MCP tools through ``scripts/mcp_call.py`` and gets
JSON back. Turning that JSON into an inventory, and turning a ``jms_dev_run`` result
into something readable, is ordinary data munging — it does not need a shell, and
doing it here keeps the shell script from growing a hand-rolled JSON parser.

Subcommands (each prints to stdout, exits nonzero on a problem):

    assets   --file assets.json          one line per visible asset
    mkhosts  --assets F --out F --login L --account A [--pick S] [--hosts S]
    discover --file discover.json --hosts hosts.yaml [--append]
    devrun   --file devrun.json

``--pick`` selects assets by index, name, or id (comma-separated). ``--hosts`` takes
``name=ref[:platform[:account]]`` entries and overrides ``--pick``.
"""

from __future__ import annotations

import argparse
import json
import re

KNOWN_PLATFORMS = ("ascend-910c", "cuda", "aarch64", "generic")


def _slug(text: str | None) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "host"


def _platform_of(asset: dict) -> str:
    """Best-effort platform for an asset, from its own field then its name/address."""
    declared = str(asset.get("platform") or "").lower()
    for p in KNOWN_PLATFORMS:
        if p in declared:
            return p
    hay = " ".join(str(asset.get(k) or "") for k in ("name", "address")).lower()
    if "ascend" in hay or "910" in hay:
        return "ascend-910c"
    if "4090" in hay or "cuda" in hay or "nvidia" in hay:
        return "cuda"
    if "aarch64" in hay or "arm" in hay:
        return "aarch64"
    return "generic"


def _resolve(assets: list[dict], token: str) -> dict:
    token = token.strip()
    if not token:
        raise SystemExit("empty asset reference")
    if token.isdigit():
        i = int(token)
        if not 0 <= i < len(assets):
            raise SystemExit(f"index {i} is out of range (0..{len(assets) - 1})")
        return assets[i]
    for a in assets:
        if a.get("id") == token:
            return a
    low = token.lower()
    for a in assets:
        if str(a.get("name") or "").lower() == low:
            return a
    for a in assets:
        if _slug(a.get("name")) == _slug(token):
            return a
    names = ", ".join(str(a.get("name") or a.get("id")) for a in assets)
    raise SystemExit(f"no asset matches {token!r} (visible: {names})")


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _refusal(data) -> str | None:
    """The text of a payload that is a refusal rather than the result we asked for."""
    if isinstance(data, dict) and "text" in data and "ok" not in data:
        return str(data["text"])
    return None


def cmd_assets(args: argparse.Namespace) -> int:
    data = _load(args.file)
    if refused := _refusal(data):
        raise SystemExit(f"jms_list_assets refused: {refused}")
    assets = data.get("assets") or []
    for i, a in enumerate(assets):
        plat = _platform_of(a)
        print(f"{i}\t{a.get('name')}\t{a.get('address')}\t{plat}\t{a.get('id')}")
    return 0


def cmd_mkhosts(args: argparse.Namespace) -> int:
    assets = _load(args.assets).get("assets") or []
    if not assets:
        raise SystemExit("jms_list_assets returned no assets; nothing to put in the inventory")

    chosen: list[dict] = []
    seen: set[str] = set()

    def add(name: str, asset: dict, platform: str | None, account: str | None) -> None:
        name = name if name not in seen else f"{name}-2"
        seen.add(name)
        chosen.append(
            {
                "name": name,
                "id": asset.get("id"),
                "platform": platform or _platform_of(asset),
                "account": account or args.account,
                "address": asset.get("address"),
            }
        )

    if args.hosts.strip():
        for item in args.hosts.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, ref = item.partition("=")
            parts = ref.split(":")
            asset = _resolve(assets, parts[0])
            platform = parts[1] if len(parts) > 1 and parts[1] else None
            account = parts[2] if len(parts) > 2 and parts[2] else None
            add(name.strip() or _slug(asset.get("name")), asset, platform, account)
    else:
        tokens = [t for t in args.pick.split(",") if t.strip()] if args.pick.strip() else []
        if tokens:
            for token in tokens:
                asset = _resolve(assets, token)
                add(_slug(asset.get("name")), asset, None, None)
        else:
            for asset in assets:
                add(_slug(asset.get("name")), asset, None, None)

    lines = [
        "# written by scripts/verify-onboarding.sh for the acceptance pass.",
        "# Real hostnames and asset ids — do not commit this file.",
        "hosts:",
    ]
    for c in chosen:
        lines += [
            f"  {c['name']}:",
            f"    login: {json.dumps(args.login)}",
            f"    account: {json.dumps(c['account'])}",
            f"    asset_id: {json.dumps(c['id'])}",
            f"    platform: {c['platform']}",
            "    tags: [verified]",
        ]
    lines += ["", "groups:", "  cluster: [" + ", ".join(c["name"] for c in chosen) + "]", ""]
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    for c in chosen:
        print(f"{c['name']}\t{c['platform']}\t{c['id']}")
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    data = _load(args.file)
    if refused := _refusal(data):
        print(f"jms_dev_discover refused: {refused}")
        return 1
    yaml_text = data.get("yaml") or ""
    if args.append and yaml_text:
        with open(args.hosts, "a") as fh:
            fh.write("\n# --- workspaces, discovered by jms_dev_discover ---\n")
            fh.write(yaml_text)
            if not yaml_text.endswith("\n"):
                fh.write("\n")
        print(f"appended {len(data.get('workspaces') or {})} workspace(s) to {args.hosts}")
    for w in (data.get("warnings") or []):
        print(f"warning: {w}")
    for host in (data.get("skipped") or []):
        print(f"skipped: {host}")
    return 0 if data.get("ok") else 1


def cmd_devrun(args: argparse.Namespace) -> int:
    data = _load(args.file)
    # A refusal that never became a table (a selector that matched nothing, a tool
    # error) arrives as plain text; there is nothing to tabulate, so print it and stop.
    if refused := _refusal(data):
        print(refused)
        return 1
    print(f"ok            {data.get('ok')}")
    print(f"stage         {data.get('stage', '-')}")
    if data.get("error"):
        print(f"error         {data['error']}")
    print(f"ref           {data.get('ref', '-')}")
    print(f"columns       {', '.join(data.get('columns') or []) or '(none)'}")
    rows = data.get("rows") or []
    bad = [r for r in rows if not r.get("consistent")]
    print(f"tests         {len(rows)} rows, {len(bad)} inconsistent")
    for r in bad[:20]:
        seen = {c: (o or {}).get("outcome") for c, o in r["outcomes"].items()}
        print(f"  DIVERGENT  {r['test_id']}  {seen}")
    print(f"revisions     {data.get('revisions')}")
    print(f"revisions_consistent  {data.get('revisions_consistent')}")
    if data.get("revisions_unverified"):
        print(f"revisions_unverified  {data['revisions_unverified']}")
    if data.get("failed_hosts"):
        print(f"failed_hosts  {data['failed_hosts']}")
    print(f"attestation_consistent  {data.get('attestation_consistent')}")
    if data.get("attestation_diverged"):
        print(f"attestation_diverged  {data['attestation_diverged']}")
    if data.get("attestation_unverified"):
        print(f"attestation_unverified  {data['attestation_unverified']}")
    for name, h in (data.get("hosts") or {}).items():
        mark = "ok" if h.get("ok") else f"FAIL(exit {h.get('exit_code')})"
        line = f"  {name}: {mark}"
        if h.get("stderr"):
            line += f"  {h['stderr']}"
        print(line)
    return 0 if data.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify_support.py", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("assets")
    p.add_argument("--file", required=True)
    p.set_defaults(func=cmd_assets)

    p = sub.add_parser("mkhosts")
    p.add_argument("--assets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--login", required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--pick", default="")
    p.add_argument("--hosts", default="")
    p.set_defaults(func=cmd_mkhosts)

    p = sub.add_parser("discover")
    p.add_argument("--file", required=True)
    p.add_argument("--hosts")
    p.add_argument("--append", action="store_true")
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("devrun")
    p.add_argument("--file", required=True)
    p.set_defaults(func=cmd_devrun)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

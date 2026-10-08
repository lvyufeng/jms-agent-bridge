"""Command-line entry points: onboarding and diagnosis.

``jms-bridge init`` writes a starter config + inventory, ``probe`` runs the real
data path read-only, and ``doctor`` checks every layer end to end so a broken setup
is diagnosed here rather than from inside a Claude Code session.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from .api import JumpServerAPI, JumpServerError
from .config import ConfigError, load_config
from .inventory import Inventory, InventoryError
from .platforms import PLATFORMS
from .ssh import SSHRunner

_CONFIG_TEMPLATE = """\
# jms-agent-bridge configuration
[bastion]
url = "https://bastion.example.com"
# KoKo SSH gateway port (JumpServer default 2222; some deployments use 2224)
ssh_port = 2222
# default login dimension used to build `<login>#<account>#<asset_id>@bastion`
login_user = "your-login"
# path to the SSH private key used for the data plane (optional; ssh-agent works too)
# ssh_key = "~/.ssh/id_ed25519"
# set false only for a self-signed internal certificate
verify_tls = true

# --- control plane credentials (prefer env vars for these) ---
# access_key_id = "..."       # or JMS_BRIDGE_ACCESS_KEY_ID
# access_key_secret = "..."   # or JMS_BRIDGE_ACCESS_KEY_SECRET
# private_token = "..."       # or JMS_BRIDGE_PRIVATE_TOKEN
# org_id = "..."              # if the deployment enforces multi-org
"""

_INVENTORY_TEMPLATE = """\
# Map logical host names to JumpServer assets + platform metadata.
hosts:
  ascend-910c:
    login: your-login
    account: root
    asset_id: 00000000-0000-0000-0000-000000000000
    platform: ascend-910c          # one of: ascend-910c, cuda, aarch64, generic
    env_setup: "source /path/to/env.sh"
    workdir: /path/to/workdir
    tags: [gpu, ascend, training]

  node-4090-1:
    login: your-login
    account: ubuntu
    asset_id: 11111111-1111-1111-1111-111111111111
    platform: cuda
    env_setup: "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fl"
    workdir: ~/torch-fl
    tags: [gpu, cuda, training]

groups:
  cluster: [ascend-910c, node-4090-1]
"""


def _config_path() -> Path:
    return Path("~/.config/jms-bridge/config.toml").expanduser()


def _inventory_path() -> Path:
    return Path("~/.config/jms-bridge/hosts.yaml").expanduser()


def cmd_init(args: argparse.Namespace) -> int:
    cfg_path, inv_path = _config_path(), _inventory_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    for path, template in ((cfg_path, _CONFIG_TEMPLATE), (inv_path, _INVENTORY_TEMPLATE)):
        if path.exists() and not args.force:
            print(f"exists, skipping (use --force to overwrite): {path}")
            continue
        path.write_text(template)
        print(f"wrote {path}")
    print(
        "\nNext:\n"
        "  1. edit both files (bastion url, login_user, asset ids, workdirs)\n"
        "  2. export JMS_BRIDGE_ACCESS_KEY_ID / JMS_BRIDGE_ACCESS_KEY_SECRET "
        "(or JMS_BRIDGE_PRIVATE_TOKEN)\n"
        "  3. jms-bridge doctor"
    )
    return 0


async def cmd_login(args: argparse.Namespace) -> int:
    """Exchange username+password for a private token and cache it in the config.

    JumpServer's API is authenticated with a long-lived token, but the credential a
    person actually has is a username/password. This logs in once and writes the
    resulting token into the config file, so subsequent runs (and the MCP server) need
    no password. The token is not echoed to the terminal.
    """
    import os
    import re

    cfg = load_config()
    username = args.username or cfg.login_user
    password = args.password or os.environ.get("JMS_BRIDGE_PASSWORD")
    if not username or not password:
        print(
            "error: need a username and password "
            "(pass --username/--password, or set JMS_BRIDGE_PASSWORD)",
            file=sys.stderr,
        )
        return 2

    api = JumpServerAPI(cfg)
    try:
        token = await api.login_with_password(username, password)
    except JumpServerError as exc:
        print(f"login failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await api.aclose()

    path = _config_path()
    text = path.read_text()
    line = f'session_token = "{token}"'
    if re.search(r"(?m)^\s*session_token\s*=", text):
        text = re.sub(r"(?m)^\s*session_token\s*=.*$", line, text)
    elif "[bastion]" in text:
        text = text.replace("[bastion]", f"[bastion]\n{line}", 1)
    else:
        text += f"\n[bastion]\n{line}\n"
    path.write_text(text)
    os.chmod(path, 0o600)
    print(f"logged in as {username}; token cached in {path} (mode 600)")
    return 0


async def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True

    print("== tooling ==")
    for tool in ("ssh", "rsync"):
        found = shutil.which(tool)
        print(f"  {tool:6} {'OK  ' + found if found else 'MISSING'}")
        ok = ok and bool(found)

    print("== config / inventory ==")
    try:
        cfg = load_config()
        print(f"  config    OK   {cfg.url} (ssh port {cfg.ssh_port})")
    except ConfigError as exc:
        print(f"  config    FAIL {exc}")
        return 1
    try:
        inv = Inventory.load()
        print(f"  inventory OK   {len(inv.hosts)} hosts, groups: {sorted(inv.groups)}")
        for h in inv.hosts.values():
            p = PLATFORMS.get(h.platform)
            flag = "" if p else f"  (unknown platform {h.platform!r} -> generic)"
            target = h.login_string(cfg.login_user)
            print(f"    - {h.name:20} platform={h.platform:12} target={target}{flag}")
    except InventoryError as exc:
        print(f"  inventory FAIL {exc}")
        return 1

    print("== control plane ==")
    api = JumpServerAPI(cfg)
    try:
        health = await api.ping()
        print(f"  /api/health/ OK   {health}")
    except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
        print(f"  /api/health/ FAIL {exc}")
        ok = False
    try:
        assets = await api.list_assets()
        print(f"  assets       OK   {len(assets)} visible")
    except JumpServerError as exc:
        print(f"  assets       SKIP {exc} (run `jms-bridge login` to authenticate)")
    except Exception as exc:  # noqa: BLE001 - doctor must not crash on a bad API
        print(f"  assets       FAIL {type(exc).__name__}: {exc}")
        ok = False
    finally:
        await api.aclose()

    print("== data plane ==")
    runner = SSHRunner(cfg)
    try:
        hosts = inv.resolve(args.targets)
        for h in hosts:
            res = await runner.exec_on(h, "echo jms-bridge-ok && hostname", timeout=45)
            mark = "OK  " if res.ok else "FAIL"
            print(f"  {h.name:20} {mark} {res.stdout.strip() or res.stderr.strip()[:200]}")
            ok = ok and res.ok
    except (InventoryError, Exception) as exc:  # noqa: BLE001
        print(f"  data plane   FAIL {exc}")
        ok = False
    finally:
        await runner.close()

    print("\n" + ("all checks passed" if ok else "some checks FAILED (see above)"))
    return 0 if ok else 1


async def cmd_probe(args: argparse.Namespace) -> int:
    cfg = load_config()
    inv = Inventory.load()
    runner = SSHRunner(cfg)
    try:
        for host in inv.resolve(args.targets):
            platform = host.platform_def
            res = await runner.exec_on(host, platform.probe_cmd, timeout=45, workdir=False)
            print(f"### {host.name} [{platform.name}]  exit={res.exit_code}")
            if res.ok:
                for k, v in platform.parse_probe(res.stdout).items():
                    print(f"    {k}: {v}")
            else:
                print(f"    {res.stderr.strip()[:400]}")
    finally:
        await runner.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jms-bridge", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="write starter config + inventory")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=lambda a: asyncio.run(_sync(cmd_init, a)))

    p_login = sub.add_parser("login", help="log in and cache a private token in the config")
    p_login.add_argument("--username", help="defaults to bastion.login_user")
    p_login.add_argument(
        "--password", help="prefer JMS_BRIDGE_PASSWORD over argv (argv leaks to ps)"
    )
    p_login.set_defaults(func=lambda a: asyncio.run(cmd_login(a)))

    p_doc = sub.add_parser("doctor", help="check tooling, config, API and SSH end to end")
    p_doc.add_argument("targets", nargs="*", default=["all"])
    p_doc.set_defaults(func=lambda a: asyncio.run(cmd_doctor(a)))

    p_probe = sub.add_parser("probe", help="probe hosts for accelerator + version")
    p_probe.add_argument("targets", nargs="*", default=["all"])
    p_probe.set_defaults(func=lambda a: asyncio.run(cmd_probe(a)))

    p_serve = sub.add_parser("serve", help="run the MCP server on stdio")
    p_serve.set_defaults(func=lambda a: _serve())

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, InventoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


async def _sync(fn, args):  # noqa: ANN001 - adapts a sync command to the async runner
    return fn(args)


def _serve() -> int:
    from .server import main as serve_main

    serve_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
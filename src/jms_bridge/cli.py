"""Command-line entry points: onboarding and diagnosis.

``jms-bridge init`` writes a starter config + inventory, ``probe`` runs the real
data path read-only, and ``doctor`` checks every layer end to end so a broken setup
is diagnosed here rather than from inside a Claude Code session.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import shutil
import sys
from pathlib import Path

import httpx

from . import secrets
from .api import JumpServerAPI, JumpServerError
from .config import ConfigError, config_path, inventory_path, load_config
from .inventory import Inventory, InventoryError
from .platforms import PLATFORMS
from .ssh import REACH_SENTINEL, SSHRunner

#: Printed by the ``doctor`` data-plane probe and required back in the reply — see
#: :data:`jms_bridge.ssh.REACH_SENTINEL` for why the exit code alone cannot be trusted.
_DOCTOR_SENTINEL = REACH_SENTINEL

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
# or the bastion login password, used via sshpass (prefer JMS_BRIDGE_SSH_PASSWORD)
# ssh_password = "..."
# set false only for a self-signed internal certificate
verify_tls = true
# parallel fan-out cap for the SSH data plane (default 8); raise it for a large fleet.
# Also settable with JMS_BRIDGE_MAX_PARALLEL.
# max_parallel = 8

# --- control plane credentials (prefer env vars for these) ---
# access_key_id = "..."       # or JMS_BRIDGE_ACCESS_KEY_ID
# access_key_secret = "..."   # or JMS_BRIDGE_ACCESS_KEY_SECRET
# private_token = "..."       # or JMS_BRIDGE_PRIVATE_TOKEN
# org_id = "..."              # if the deployment enforces multi-org
"""

_INVENTORY_TEMPLATE = """\
# Map logical host names to JumpServer assets + platform metadata.
#
# Each host needs a JumpServer asset id, which you cannot know before you are logged
# in. Do it in that order: fill in config.toml, run `jms-bridge login`, register the
# MCP server, call `jms_list_assets`, *then* fill in the ids below. The JumpServer web
# UI shows them too (`jms_list_assets` just saves you the lookup).
#
# `examples/hosts.yaml` in the repo is this same file with every field exercised.
hosts:
  ascend-910c:
    login: your-login
    account: root
    asset_id: 00000000-0000-0000-0000-000000000000
    platform: ascend-910c          # one of: ascend-910c, cuda, aarch64, generic,
                                   #         hygon-dcu, metax, enflame, kunlunxin,
                                   #         thead, iluvatar, moore-threads, tsingmicro
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

# Workspaces are what the `jms_dev_*` development loop operates on: a host plus the
# *container* on it that holds your checkout. Without a workspace here, `jms_dev_run`
# matches nothing (`workspaces: []`) and cannot deliver code to any chip.
#
#   container  the same name on every host (must match [a-zA-Z0-9][a-zA-Z0-9_.-]*)
#   repo       path to the checkout *inside* that container
#   source     git | proxy-git | push   (how a ref reaches the container)
#              proxy-git also needs `proxy:`, push also needs `mount:` + `local:`
#   env_setup  sourced *inside* the container before the checkout and the UT command
#   ut_cmd     must emit pytest -rA short-summary lines (the default already does)
#   tags       free-form; usable as selectors, like host tags
#
# `jms_dev_discover` fills this in for you against a live cluster — prefer that once
# the hosts above work. The block below is the shape it emits, reduced to one entry.
workspaces:
  fl-ascend:
    host: ascend-910c
    container: your-torch-fl
    repo: /work/Torch-FL
    source: git
    env_setup: "source /usr/local/Ascend/ascend-toolkit/set_env.sh && conda activate fl"
    tags: [ascend]
"""


def _config_path() -> Path:
    # Delegates to config.config_path so the CLI and the config loader agree on
    # precedence (JMS_BRIDGE_CONFIG wins) — otherwise `login` would cache a token in
    # the default file while the server read a different one.
    return config_path()


def _inventory_path() -> Path:
    return inventory_path()


def cmd_init(args: argparse.Namespace) -> int:
    cfg_path, inv_path = _config_path(), _inventory_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    for path, template in ((cfg_path, _CONFIG_TEMPLATE), (inv_path, _INVENTORY_TEMPLATE)):
        if path.exists() and not args.force:
            print(f"exists, skipping (use --force to overwrite): {path}")
            continue
        path.write_text(template)
        # config.toml will hold the session token after `login`, and hosts.yaml names
        # every asset and login; keep both owner-only from the start rather than relying
        # on a later `login` (which only tightens the config).
        os.chmod(path, 0o600)
        print(f"wrote {path}")
    print(
        "\nNext:\n"
        "  1. edit config.toml (bastion url, login_user) and hosts.yaml (hostnames,\n"
        "     workdirs, the container in `workspaces:`). Leave the asset ids for step 3.\n"
        "  2. jms-bridge login --save-password\n"
        "     (prompts for the password; stores it in the OS keyring, else a 0600 file,\n"
        "      so later logins need no secret — or export JMS_BRIDGE_PASSWORD instead)\n"
        "     (if your deployment issues API keys:\n"
        "      export JMS_BRIDGE_ACCESS_KEY_ID / JMS_BRIDGE_ACCESS_KEY_SECRET,\n"
        "      or JMS_BRIDGE_PRIVATE_TOKEN)\n"
        "  3. jms-bridge doctor — tooling, config and control plane should pass; the\n"
        "     data plane cannot until step 4 gives it real asset ids.\n"
        "  4. asset ids: you need to be logged in to look them up. Register the MCP\n"
        "     server (see README) and call jms_list_assets, or read them off the\n"
        "     JumpServer web UI, then fill in `asset_id` above and re-run doctor\n"
        "     (and `jms-bridge probe cluster`).\n"
        "     The data plane needs its own credential: an SSH key registered with\n"
        "     JumpServer (recommended), or JMS_BRIDGE_SSH_PASSWORD plus sshpass."
    )
    return 0


async def cmd_login(args: argparse.Namespace) -> int:
    """Exchange username+password for a session token and cache it in the config.

    JumpServer's API is authenticated with a long-lived token, but the credential a
    person actually has is a username/password. This logs in once and writes the
    resulting token into the config file, so subsequent runs (and the MCP server) need
    no password. The token is not echoed to the terminal.

    The password is read from ``--password``, then ``JMS_BRIDGE_PASSWORD``, then a
    previously saved copy (keyring, else the 0600 credentials file), and only then
    prompted for. ``--save-password`` stores it for next time; ``--forget`` deletes it.
    """
    import re

    if getattr(args, "forget", False):
        cleared = secrets.forget_password()
        print(
            "forgot the saved password (" + ", ".join(cleared) + ")"
            if cleared
            else "no saved password to forget"
        )
        return 0

    cfg = load_config()
    username = args.username or cfg.login_user
    password = args.password or secrets.get_password()
    if not password and sys.stdin.isatty():
        password = getpass.getpass(f"JumpServer password for {username or 'login'}: ")
    if not username or not password:
        print(
            "error: need a username and password (pass --username/--password, set "
            "JMS_BRIDGE_PASSWORD, or run `jms-bridge login --save-password` once to "
            "store it for later logins)",
            file=sys.stderr,
        )
        return 2

    api = JumpServerAPI(cfg)
    try:
        token = await api.login_with_password(username, password)
    except (JumpServerError, httpx.HTTPError) as exc:
        # httpx failures (DNS, refused connection, TLS, timeout) are the *expected*
        # first-run outcome for a mistyped bastion URL; report them as a message
        # rather than a traceback.
        print(f"login failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await api.aclose()

    path = _config_path()
    if not path.exists():
        print(
            f"error: no config at {path}.\n"
            "Run `jms-bridge init` first (or set JMS_BRIDGE_CONFIG to your config file), "
            "then edit `bastion.url` before logging in.",
            file=sys.stderr,
        )
        return 2
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
    if getattr(args, "save_password", False):
        print(f"saved the password to {secrets.save_password(password)}")
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

    print("== credentials ==")
    # Two different secrets, deliberately named differently: the control plane takes the
    # JumpServer *login* password (or a token), the data plane takes what the KoKo
    # gateway needs. `login --save-password` only covers the control plane, so label
    # them here rather than let a user assume one covers both.
    print(
        f"  JMS_BRIDGE_PASSWORD      (control plane) "
        f"{'set' if os.environ.get('JMS_BRIDGE_PASSWORD') else 'unset'}"
    )
    print(
        f"  JMS_BRIDGE_SSH_PASSWORD  (data plane)    "
        f"{'set' if os.environ.get('JMS_BRIDGE_SSH_PASSWORD') else 'unset'}"
    )
    saved = secrets.stored_password_source()
    print(f"  saved password       {saved or 'none (store: jms-bridge login --save-password)'}")
    backend = secrets.keyring_backend() or f"unavailable ({secrets.KEYRING_HINT})"
    print(f"  keyring backend      {backend}")
    if cfg.ssh_password:
        # sshpass is only needed for password auth, and the ssh/rsync check above cannot
        # know that yet — report it here so the gate is not "tooling all OK" followed by
        # an unexplained data-plane failure.
        found = shutil.which("sshpass")
        print(f"  sshpass              {'OK  ' + found if found else 'MISSING'}")
        if not found:
            print(
                "                       ssh_password is set but sshpass is not installed; "
                "install it (apt/brew) or register an SSH key instead"
            )
            ok = False
    else:
        auth = f"ssh_key {cfg.ssh_key}" if cfg.ssh_key else "ssh-agent / default keys"
        print(f"  data plane auth      {auth}")

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
    runner = SSHRunner(cfg, max_parallel=cfg.max_parallel)
    try:
        hosts = inv.resolve(args.targets)
        for h in hosts:
            res = await runner.exec_on(h, f"echo {_DOCTOR_SENTINEL} && hostname", timeout=45)
            # Require the sentinel, not just a zero exit: a KoKo rejection exits 0 with
            # the error on stdout, and trusting that would report a dead asset green.
            reached = res.ok and _DOCTOR_SENTINEL in res.stdout
            mark = "OK  " if reached else "FAIL"
            print(f"  {h.name:20} {mark} {res.stdout.strip() or res.stderr.strip()[:200]}")
            ok = ok and reached
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
    runner = SSHRunner(cfg, max_parallel=cfg.max_parallel)
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

    p_login = sub.add_parser(
        "login", help="log in with username/password and cache a session token in the config"
    )
    p_login.add_argument("--username", help="defaults to bastion.login_user")
    p_login.add_argument(
        "--password", help="prefer JMS_BRIDGE_PASSWORD over argv (argv leaks to ps)"
    )
    p_login.add_argument(
        "--save-password",
        action="store_true",
        help="store the password (OS keyring, else a 0600 file) so later logins need no secret",
    )
    p_login.add_argument(
        "--forget", action="store_true", help="delete the saved password and exit"
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
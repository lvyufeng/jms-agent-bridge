"""Discovery: turn a list of hosts into a paste-ready ``workspaces:`` block.

The flagos dev loop needs three facts per chip before it can fan out a UT run:
*which* container holds the repo, *where* the checkout lives inside it, and
*whether* the container can reach the git remote. Gathering those by hand — one
``docker exec`` per host with two levels of shell quoting — is exactly the kind
of per-host busywork the bridge exists to remove. This module does it in one call
and emits inventory YAML the developer pastes into ``hosts.yaml``.

Two shell layers are involved and the quoting is the whole game:

* the *host* layer, where the bridge's own wrapper already runs ``set -euo
  pipefail`` — so the script starts by relaxing that (a missing container, an
  absent ``find`` root or an unreachable remote must not abort discovery); and
* the *container* layer, reached with ``docker exec <c> bash -lc <script>``.

The container's script is single-quoted with :func:`~jms_bridge.inventory.shell_quote`
so the host shell never expands the container's own ``$vars`` — the bug that made
an earlier hand-rolled version die with ``R: unbound variable``. Nothing in the
container script uses ``${...}`` braces, so it also survives being embedded in an
f-string here.

The container name is the one value interpolated into the command; it is validated
against :data:`jms_bridge.workspace._CONTAINER_RE` before use.
"""

from __future__ import annotations

import yaml

from .inventory import shell_quote

#: Where a flagos checkout is likely to live inside a batch-node container.
DEFAULT_ROOTS = ["/public-flash", "/work", "/workspace", "/root", "/home", "/opt"]

#: A discovered checkout is only reported when its path contains this, matched
#: case-insensitively. ``torch`` catches both ``PyTorch-Plugin-FL`` (the real
#: checkout) and a ``torch-fl`` layout; over-matching is acceptable because every
#: match is reported for the developer to confirm.
DEFAULT_REPO_MATCH = "torch"

#: Key that prefixes every machine-readable line the discovery script prints.
MARKER = "JMSD "

#: The script that runs *inside* the container. No ``${...}`` braces (so it can be
#: embedded in an f-string) and no reliance on the outer ``set -u``. ``$pv`` may be
#: unset when pytest is absent — hence the relaxed shell, and ``"$pv"`` not ``${pv:-}``.
_INNER_TEMPLATE = """\
set +eu +o pipefail
printf 'JMSD inner=1\\n'
for p in python3 python3.13 python3.12 python3.11 python3.10; do
  command -v "$p" >/dev/null 2>&1 || continue
  v=$("$p" -c "import sys;print(sys.version.split()[0])" 2>/dev/null)
  [ -n "$v" ] || continue
  pv=$("$p" -c "import pytest;print(pytest.__version__)" 2>/dev/null)
  printf 'JMSD python=%s|%s|%s\\n' "$p" "$v" "$pv"
done
ML=$(printf '%s' {match} | tr 'A-Z' 'a-z')
for r in {roots}; do
  [ -d "$r" ] || continue
  for d in $(find "$r" -maxdepth {depth} -name .git -type d 2>/dev/null); do
    d=$(dirname "$d")
    dl=$(printf '%s' "$d" | tr 'A-Z' 'a-z')
    case "$dl" in
      *"$ML"*) ;;
      *) continue ;;
    esac
    url=$(git -C "$d" remote get-url origin 2>/dev/null)
    br=$(git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null)
    sha=$(git -C "$d" rev-parse HEAD 2>/dev/null)
    rc=0
    if command -v timeout >/dev/null 2>&1; then
      timeout {reach} git -C "$d" ls-remote origin HEAD >/dev/null 2>&1 || rc=1
    else
      git -C "$d" ls-remote origin HEAD >/dev/null 2>&1 || rc=1
    fi
    printf 'JMSD repo=%s\\n' "$d"
    printf 'JMSD remote=%s\\n' "$url"
    printf 'JMSD branch=%s\\n' "$br"
    printf 'JMSD sha=%s\\n' "$sha"
    printf 'JMSD reachable=%s\\n' "$rc"
  done
done
"""

#: The script that runs on the *host*: locate the container, delegate to the inner
#: script, and never let a missing container fail the call (discovery is a report).
_HOST_TEMPLATE = """\
set +eu +o pipefail
C={container}
if ! command -v docker >/dev/null 2>&1; then
  printf 'JMSD error=docker_absent\\n'
  exit 0
fi
if ! docker inspect "$C" >/dev/null 2>&1; then
  printf 'JMSD container_absent=1\\n'
  exit 0
fi
printf 'JMSD container=%s\\n' "$C"
if docker ps -q --filter "name=^$C$" | grep -q .; then
  printf 'JMSD running=1\\n'
else
  printf 'JMSD running=0\\n'
fi
docker exec "$C" bash -lc {inner} || printf 'JMSD inner_exit=%s\\n' "$?"
exit 0
"""


def build_discovery_script(
    container: str,
    *,
    roots: list[str] | None = None,
    repo_match: str = DEFAULT_REPO_MATCH,
    max_depth: int = 5,
    reach_timeout: int = 8,
) -> str:
    """Build the host-side script that discovers one container's workspace facts."""
    inner = _INNER_TEMPLATE.format(
        match=shell_quote(repo_match),
        roots=" ".join(shell_quote(r) for r in (roots or DEFAULT_ROOTS)),
        depth=max_depth,
        reach=reach_timeout,
    )
    return _HOST_TEMPLATE.format(container=shell_quote(container), inner=shell_quote(inner))


def parse_discovery(stdout: str) -> dict:
    """Parse the ``JMSD ``-prefixed lines of a discovery run into a report.

    A ``repo=`` line starts a new checkout record; the ``remote/branch/sha/
    reachable`` lines that follow attach to it (fields with no owning record are
    dropped rather than silently misattributed). Field values run to end-of-line,
    so a path may contain spaces.
    """
    report: dict = {
        "container_present": False,
        "container": None,
        "running": False,
        "inner_ok": False,
        "python": [],
        "repos": [],
        "error": None,
    }
    repo: dict | None = None
    for line in stdout.splitlines():
        line = line.rstrip()
        if not line.startswith(MARKER):
            continue
        body = line[len(MARKER):]
        key, _, value = body.partition("=")
        if body == "container_absent=1":
            continue  # container_present stays False
        if key == "container":
            report["container"] = value
            report["container_present"] = True
        elif key == "running":
            report["running"] = value == "1"
        elif key == "inner":
            report["inner_ok"] = value == "1"
        elif key == "error":
            report["error"] = value
        elif key == "python":
            path, _, rest = value.partition("|")
            version, _, pytest_version = rest.partition("|")
            report["python"].append(
                {"path": path, "version": version, "pytest": pytest_version or None}
            )
        elif key == "repo":
            repo = {"path": value, "remote": None, "branch": None, "sha": None, "reachable": None}
            report["repos"].append(repo)
        elif key in ("remote", "branch", "sha", "reachable") and repo is not None:
            repo[key] = (value == "1") if key == "reachable" else (value or None)
    return report


def _workspace_key(host: str, index: int) -> str:
    return f"fl-{host}" if index == 0 else f"fl-{host}-{index + 1}"


def render_workspaces_yaml(
    reports: list[dict],
    *,
    container: str,
    source: str = "git",
    host_tags: dict[str, list[str]] | None = None,
) -> dict:
    """Turn per-host discovery reports into a paste-ready ``workspaces:`` document.

    Each host contributes one workspace per discovered checkout (a host with two
    checkouts gets ``fl-<host>`` and ``fl-<host>-2``). Hosts without the container
    or without a matching checkout are reported in ``skipped`` — never emitted as a
    half-formed entry that would only fail later at delivery time. Every caveat
    (a container the tool could not inspect, an origin the container cannot reach,
    a missing pytest) becomes a ``warnings`` entry, because the emitted ``source``
    is a *guess* the developer must confirm.
    """
    host_tags = host_tags or {}
    workspaces: dict[str, dict] = {}
    warnings: list[str] = []
    skipped: list[str] = []

    for item in reports:
        host = item["host"]
        rep = item["report"]
        if not rep["container_present"]:
            warnings.append(f"{host}: container {container!r} not found")
            skipped.append(host)
            continue
        if not rep["inner_ok"]:
            warnings.append(
                f"{host}: could not run the discovery script inside {container!r}"
                + (f" ({rep['error']})" if rep.get("error") else "")
            )
        if not rep["repos"]:
            warnings.append(f"{host}: no matching checkout found inside {container!r}")
            skipped.append(host)
            continue
        if len(rep["repos"]) > 1:
            warnings.append(
                f"{host}: {len(rep['repos'])} checkouts matched — emitting one workspace each"
            )
        if not any(p["pytest"] for p in rep["python"]):
            warnings.append(
                f"{host}: no python interpreter with pytest found — set 'ut_cmd' for this workspace"
            )

        tags = list(host_tags.get(host) or [])
        for i, repo in enumerate(rep["repos"]):
            entry: dict = {
                "host": host,
                "container": container,
                "repo": repo["path"],
                "source": source,
            }
            if repo["reachable"] != 1:
                warnings.append(
                    f"{host}: origin unreachable from the container — confirm "
                    f"'source: {source}'; if it stays offline use 'proxy-git' "
                    "(+ 'proxy:') or 'push' (+ 'mount:' and 'local:')"
                )
            if tags:
                entry["tags"] = tags
            workspaces[_workspace_key(host, i)] = entry

    document = {"workspaces": workspaces}
    yaml_text = (
        "# generated by jms_dev_discover — review before use\n"
        "# select with jms_dev_run([\"all\"], ref=...) or by tag\n"
        + yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
    )
    return {
        "workspaces": workspaces,
        "yaml": yaml_text,
        "warnings": warnings,
        "skipped": skipped,
    }
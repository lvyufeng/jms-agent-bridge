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
against :data:`jms_bridge.workspace.CONTAINER_RE` before use.

Every failure mode is reported as itself, not as a neighbouring one: a host with
no Docker, a container that is not there, a ``docker exec`` that failed, a probe
that timed out and a checkout that does not exist are five distinct outcomes. A
misdiagnosis here sends the developer to fix the wrong thing, so the renderer
carries the raw exit status and the specific error through to the warning text.
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

#: ``reachable=`` values. The probe deliberately keeps three outcomes apart, because
#: "could not tell" (a probe that timed out) and "cannot reach" (an auth/DNS/route
#: failure) want different advice — the first says retry, the second says change
#: ``source``.
_REACH_OK = "ok"
_REACH_DENIED = "denied"
_REACH_TIMEOUT = "timeout"


#: The script that runs *inside* the container. No ``${...}`` braces (so it can be
#: embedded in an f-string) and no reliance on the outer ``set -u``. ``find -print0``
#: + ``read -d ''`` keeps a checkout path with spaces or a glob metacharacter intact
#: (an unquoted ``$(find ...)`` word-splits and glob-expands it into garbage paths).
#: ``$pv`` may be unset when pytest is absent — hence the relaxed shell.
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
  find "$r" -maxdepth {depth} -name .git -type d -print0 2>/dev/null \
    | while IFS= read -r -d '' g; do
    d=$(dirname "$g")
    dl=$(printf '%s' "$d" | tr 'A-Z' 'a-z')
    case "$dl" in
      *"$ML"*) ;;
      *) continue ;;
    esac
    url=$(git -C "$d" remote get-url origin 2>/dev/null)
    br=$(git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null)
    sha=$(git -C "$d" rev-parse HEAD 2>/dev/null)
    rc={reach_ok}
    if command -v timeout >/dev/null 2>&1; then
      timeout {reach} git -C "$d" ls-remote origin HEAD >/dev/null 2>&1
      t=$?
      if [ "$t" -eq 124 ]; then rc={reach_timeout}; elif [ "$t" -ne 0 ]; then rc={reach_denied}; fi
    else
      git -C "$d" ls-remote origin HEAD >/dev/null 2>&1 || rc={reach_denied}
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
#: A failed ``docker exec`` prints its exit status so the renderer can say *why*
#: the container could not be inspected instead of guessing.
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
if docker exec "$C" bash -lc {inner}; then :; else printf 'JMSD inner_exit=%s\\n' "$?"; fi
exit 0
"""


def build_discovery_script(
    container: str,
    *,
    roots: list[str] | None = None,
    repo_match: str = DEFAULT_REPO_MATCH,
    max_depth: int = 5,
    reach_timeout: int = 30,
) -> str:
    """Build the host-side script that discovers one container's workspace facts.

    ``reach_timeout`` bounds each ``git ls-remote`` probe. It is deliberately
    generous: a git remote that cannot be reached often takes ~25-30s to fail
    (connect retries), and a shorter budget would turn an actionable "unreachable /
    unauthenticated" into an unhelpful "could not determine".
    """
    """Build the host-side script that discovers one container's workspace facts."""
    inner = _INNER_TEMPLATE.format(
        match=shell_quote(repo_match),
        roots=" ".join(shell_quote(r) for r in (roots or DEFAULT_ROOTS)),
        depth=max_depth,
        reach=reach_timeout,
        reach_ok=_REACH_OK,
        reach_denied=_REACH_DENIED,
        reach_timeout=_REACH_TIMEOUT,
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
        "inner_exit": None,
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
        elif key == "inner_exit":
            report["inner_exit"] = int(value) if value.isdigit() else None
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
            repo[key] = value or None
    return report


def _base_key(host: str, index: int) -> str:
    return f"fl-{host}" if index == 0 else f"fl-{host}-{index + 1}"


def _unique_key(base: str, used: set[str]) -> str:
    """``base``, suffixed until it is not already taken.

    A host with two checkouts produces ``fl-node-2``; a *different* host literally
    named ``node-2`` produces ``fl-node-2`` too. Without this the second write
    silently overwrote the first and a discovered workspace vanished.
    """
    key = base
    i = 2
    while key in used:
        key = f"{base}-{i}"
        i += 1
    return key


def render_workspaces_yaml(
    reports: list[dict],
    *,
    container: str,
    source: str = "git",
    proxy: str | None = None,
    mount: str | None = None,
    local: str | None = None,
    host_tags: dict[str, list[str]] | None = None,
) -> dict:
    """Turn per-host discovery reports into a paste-ready ``workspaces:`` document.

    Each host contributes one workspace per discovered checkout (a host with two
    checkouts gets ``fl-<host>`` and ``fl-<host>-2``). Hosts that could not be
    inspected at all — no Docker, no container, a failed ``docker exec``, a
    transport timeout, or no matching checkout — are reported in ``skipped`` with a
    warning that names the actual fault, never emitted as a half-formed entry that
    would only fail later at delivery time.

    ``proxy`` / ``mount`` / ``local`` are threaded into the emitted entries so that
    a ``proxy-git`` or ``push`` workspace is complete: without them the block would
    load-fail the *whole* inventory (``Workspace.__post_init__`` requires them),
    which is why the caller validates them up front.
    """
    host_tags = host_tags or {}
    workspaces: dict[str, dict] = {}
    warnings: list[str] = []
    skipped: list[str] = []

    for item in reports:
        host = item["host"]
        rep = item["report"]
        # Order matters: name the *most specific* fault first, so a timeout is never
        # reported as a missing container, and a Docker-less host is not sent on a
        # hunt for a container that was never the problem.
        if item.get("timed_out"):
            warnings.append(
                f"{host}: discovery timed out after {item.get('timeout')}s "
                "— the host is slow or unreachable, retry or raise the timeout"
            )
            skipped.append(host)
            continue
        if not rep["container_present"]:
            if rep.get("error") == "docker_absent":
                warnings.append(
                    f"{host}: no docker CLI on the host (or not on PATH) — cannot inspect "
                    f"{container!r}"
                )
            else:
                warnings.append(f"{host}: container {container!r} not found")
            skipped.append(host)
            continue
        if not rep["inner_ok"]:
            if rep.get("inner_exit") is not None:
                detail = f"docker exec exited {rep['inner_exit']}"
            elif rep.get("error"):
                detail = rep["error"]
            else:
                detail = "no output"
            warnings.append(f"{host}: could not inspect container {container!r} ({detail})")
            skipped.append(host)
            continue
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
            if source == "proxy-git" and proxy:
                entry["proxy"] = proxy
            if source == "push":
                if mount:
                    entry["mount"] = mount
                if local:
                    entry["local"] = local
            reach = repo["reachable"]
            if reach == _REACH_TIMEOUT:
                warnings.append(
                    f"{host}: could not determine whether {repo['path']} can reach its origin "
                    "(probe timed out) — confirm 'source' before relying on it"
                )
            elif reach == _REACH_DENIED:
                warnings.append(
                    f"{host}: origin of {repo['path']} did not answer (as unreachable *or* "
                    f"unauthenticated) — confirm 'source: {source}'; if it stays offline use "
                    "'proxy-git' (+ 'proxy:') or 'push' (+ 'mount:' and 'local:')"
                )
            elif reach != _REACH_OK:
                warnings.append(
                    f"{host}: origin reachability of {repo['path']} could not be determined"
                )
            if tags:
                entry["tags"] = tags
            key = _unique_key(_base_key(host, i), set(workspaces))
            workspaces[key] = entry

    document = {"workspaces": workspaces}
    yaml_text = (
        "# generated by jms_dev_discover — review before use\n"
        "# select with jms_dev_run([\"all\"], ref=...) or by tag\n"
        + yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
    )
    return {
        # A clean discovery is one where every requested host produced a workspace;
        # a soft warning (an offline origin, a missing pytest) is a caution, not a
        # failure, but a skipped host means the picture is incomplete.
        "ok": bool(workspaces) and not skipped,
        "workspaces": workspaces,
        "yaml": yaml_text,
        "warnings": warnings,
        "skipped": skipped,
    }
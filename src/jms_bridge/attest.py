"""Attestation: record *what ran*, so a cross-chip PASS means the same code path.

The consistency table (``workspace.build_consistency_table``) compares outcomes only, and
``jms_dev_run`` gates ``ok`` on delivery, exit code and a shared git ``revisions`` sha.
Neither observes what actually executed. Two chips at the same sha but different builds —
or dispatching the same op to a different backend — both report ``PASS``, and the table
asserts uniform behaviour it never saw.

This module runs one in-container probe and turns it into an attestation record.

The probe has **two layers**, because the container's interpreter frequently cannot import
the plugin at all (measured on ``ascend-910c-108``: the ``_C`` extension is built for
cpython-310/314 and the default interpreter is 3.11, so ``import torch_fl`` dies with
``ModuleNotFoundError``). So:

* **static** — read the checkout directly: ``torch_fl/_build_config.py`` (the
  ``ACCELERATOR`` the wheel was built for; the maintainers generate it precisely to tell
  wheels apart), ``torch_fl/backends.conf`` (the ``op -> backend`` map — the *code path*),
  and the ``_C`` extension's ABI tag vs the interpreter's; and
* **runtime** — best-effort ``import torch_fl`` for its version/backend, recorded as an
  ``import_error`` when it fails rather than swallowed.

Every line is prefixed with :data:`MARKER` so the record survives being interleaved with
pytest output in the same ``jms_dev_run`` round trip.
"""

from __future__ import annotations

import hashlib
import re

from .inventory import shell_quote

#: Line prefix for every machine-readable attestation fact.
MARKER = "JMSA "

#: Interpreter the runtime layer uses inside the container.
DEFAULT_PYTHON = "python3"

#: ``ACCELERATOR = "ascend"`` as setup.py writes it into _build_config.py.
_ACCEL_RE = re.compile(r"""ACCELERATOR\s*=\s*["']([^"']+)["']""")

#: An ``op = backend`` assignment. The value may carry spaces and is allowed a
#: trailing ``# comment`` (both are stripped); a line with no ``=`` is not a map entry.
_BACKEND_RE = re.compile(r"^\s*([^\s#=][^=]*?)\s*=\s*([^#]*?)\s*(?:#.*)?$")


def build_attest_script(repo: str, python: str = DEFAULT_PYTHON) -> str:
    """Build the in-container snippet that attests one checkout.

    Composable on purpose: the caller appends it to the UT command so the record rides
    the *same* SSH round trip ``jms_dev_run`` already makes. So it must be safe to drop
    after any shell state — hence the opening ``set +eu +o pipefail`` and the total
    absence of ``exit`` (an ``exit`` here would clobber the UT run's exit status). Every
    failure, including "could not ``cd``", is a *recorded* fact, because a report must
    not abort on the very condition it exists to surface.

    Assembled by concatenation (not ``str.format``) so the runtime python block's braces
    need no escaping.
    """
    return (
        "set +eu +o pipefail\n"
        f"if cd {shell_quote(repo)} 2>/dev/null; then\n"
        # --- static: build identity ---
        "  if [ -f torch_fl/_build_config.py ]; then\n"
        "    grep -m1 ACCELERATOR torch_fl/_build_config.py "
        "| sed 's/^/JMSA build_config=/' 2>/dev/null\n"
        "  else printf 'JMSA build_config_absent=1\\n'; fi\n"
        # --- static: op -> backend map (the code path) ---
        # Emit an op= line even when the file is empty/comment-only: the *presence*
        # of the map is a fact, and a zero-entry map that cannot be told from an
        # unreadable one would leave a legitimate chip permanently `unverified`.
        "  if [ -f torch_fl/backends.conf ]; then\n"
        "    printf 'JMSA backends_read=1\\n'\n"
        "    grep -vE '^[[:space:]]*(#|$)' torch_fl/backends.conf "
        "| while IFS= read -r ln; do printf 'JMSA backend=%s\\n' \"$ln\"; done\n"
        "  else printf 'JMSA backends_absent=1\\n'; fi\n"
        # --- static: extension ABI vs interpreter ABI ---
        # Extract `cpython-3NN` from `_C.cpython-310-aarch64-linux-gnu.so`. A shell
        # `${a%%-*}` would stop at the *first* dash and yield only `cpython`, so use an
        # anchored sed on the basename instead.
        "  so=$(ls torch_fl/_C.cpython-*.so 2>/dev/null | head -1)\n"
        "  if [ -n \"$so\" ]; then\n"
        "    a=$(basename \"$so\" | sed -n 's/^_C\\.\\(cpython-[0-9]*\\).*/\\1/p')\n"
        "    [ -n \"$a\" ] && printf 'JMSA so_abi=%s\\n' \"$a\"\n"
        "  fi\n"
        f"  PY={shell_quote(python)}\n"
        "  printf 'JMSA python=%s\\n' \"$PY\"\n"
        "  ia=$(\"$PY\" -c \"import sys;print('cpython-%d%d'%sys.version_info[:2])\" "
        "2>/dev/null)\n"
        "  [ -n \"$ia\" ] && printf 'JMSA interp_abi=%s\\n' \"$ia\"\n"
        # --- runtime: best-effort import, bounded ---
        # ``$TO`` word-splits to expand to two args on purpose; an unbounded import can
        # hang on a wedged driver, which would take the whole UT run down with it.
        "  TO=''; command -v timeout >/dev/null 2>&1 && TO='timeout 30'\n"
        "  TORCH_DEVICE_BACKEND_AUTOLOAD=0 $TO \"$PY\" - <<'JMSA_PY'\n"
        "import json\n"
        "out = {}\n"
        "try:\n"
        "    import torch\n"
        "    out['torch'] = torch.__version__\n"
        "except Exception:\n"
        "    out['torch'] = None\n"
        "try:\n"
        "    import torch_fl\n"
        "    out['plugin_version'] = getattr(torch_fl, '__version__', None)\n"
        "    try:\n"
        "        out['runtime_accelerator'] = torch_fl._build_accelerator()\n"
        "    except Exception:\n"
        "        out['runtime_accelerator'] = None\n"
        "except Exception as e:\n"
        "    out['import_error'] = type(e).__name__ + ': ' + str(e)\n"
        "print('JMSA runtime=' + json.dumps(out))\n"
        "JMSA_PY\n"
        "else\n"
        "  printf 'JMSA error=cd_failed\\n'\n"
        "fi\n"
    )


def _hash_map(mapping: dict[str, str]) -> str:
    """The canonical digest of an ``op -> backend`` map.

    An empty map hashes too (to the digest of the empty string): "read, and it defines
    no entries" is a *fact* two chips can agree on, distinct from "could not be read"
    (which the caller represents as ``None`` by not calling this at all).
    """
    blob = "\n".join(f"{k}={mapping[k]}" for k in sorted(mapping))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def parse_attestation(stdout: str) -> dict:
    """Parse ``JMSA ``-prefixed lines into a per-workspace attestation record.

    ``backend_map`` is the parsed ``op -> backend`` map and ``backend_map_hash`` its
    canonical digest — the value two chips must agree on for the same *code path*.
    ``abi_match`` is ``so_abi == interp_abi`` when both are known (else ``None``): it
    explains an import failure ("the extension is cpython-310, the interpreter is 3.11")
    instead of leaving it mysterious.
    """
    rec: dict = {
        "accelerator": None,       # build identity, from _build_config.py
        "backend_map": {},
        "backend_map_hash": None,
        #: Whether `backends.conf` was *read*. Distinguishes a genuinely empty map
        #: (read, hash of "") from an unreadable one (absent) — without this both
        #: collapse to "no hash" and a legitimately-empty chip reads as unverified.
        "backends_read": False,
        "so_abi": None,
        "interp_abi": None,
        "python": None,
        "torch": None,
        "plugin_version": None,
        "runtime_accelerator": None,
        "import_error": None,
        "missing": [],
    }
    runtime: dict = {}
    for line in stdout.splitlines():
        if not line.startswith(MARKER):
            continue
        body = line[len(MARKER):].rstrip("\n")
        key, _, value = body.partition("=")
        if key == "build_config":
            m = _ACCEL_RE.search(value)
            if m:
                rec["accelerator"] = m.group(1)
        elif key == "backend":
            m = _BACKEND_RE.match(value)
            if m:
                rec["backend_map"][m.group(1).strip()] = m.group(2).strip()
        elif key in ("so_abi", "interp_abi", "python"):
            rec[key] = value or None
        elif key == "backends_read":
            rec["backends_read"] = True
        elif key == "runtime":
            try:
                import json as _json

                runtime = _json.loads(value)
            except ValueError:
                runtime = {}
            rec["torch"] = runtime.get("torch")
            rec["plugin_version"] = runtime.get("plugin_version")
            rec["runtime_accelerator"] = runtime.get("runtime_accelerator")
            rec["import_error"] = runtime.get("import_error")
        elif key.endswith("_absent"):
            rec["missing"].append(key[: -len("_absent")])
        elif key == "error":
            rec["missing"].append(value)

    rec["backend_map_hash"] = _hash_map(rec["backend_map"]) if rec["backends_read"] else None
    if rec["accelerator"] is None and rec["runtime_accelerator"]:
        # The static read failed but the import worked — take the runtime's answer.
        rec["accelerator"] = rec["runtime_accelerator"]
    if rec["so_abi"] and rec["interp_abi"]:
        rec["abi_match"] = rec["so_abi"] == rec["interp_abi"]
    else:
        rec["abi_match"] = None
    return rec


#: Fields that must agree across every chip for the table's rows to be comparable — a
#: shared *code path*, not merely a shared git sha. ``backend_map_hash`` is the op
#: dispatch; ``plugin_version`` (when known everywhere) is the plugin both ran. The build
#: ``accelerator``, the ABI and the installed ``torch`` legitimately differ per chip and
#: are surfaced, not gated.
def check_attestation_parity(per_workspace: dict[str, dict]) -> dict:
    """Compare per-workspace attestations, returning the fields that diverge.

    ``backend_map_hash`` is gated only among the workspaces that reported a map — a chip
    whose map could not be read is ``unverified``, never coerced into agreement. Returns
    ``{"consistent": bool, "diverged": [...], "unverified": [...]}``.
    """
    if not per_workspace:
        return {"consistent": False, "diverged": ["no workspaces"], "unverified": []}

    # A map that was *read* is gated, even if it is empty (a chip whose checkout
    # legitimately defines no op->backend entries must not read as unverified). Only
    # a chip whose backends.conf could not be read at all (`backends_read` False,
    # hence no hash) is unverified.
    unverified = [name for name, rec in per_workspace.items() if not rec.get("backends_read")]
    gated = {
        name: rec["backend_map_hash"]
        for name, rec in per_workspace.items()
        if rec.get("backends_read")
    }
    diverged: list[str] = []
    if len(set(gated.values())) > 1:
        diverged.append("backend_map_hash")
    versions = {rec.get("plugin_version") for rec in per_workspace.values()}
    versions.discard(None)
    if len(versions) > 1:
        diverged.append("plugin_version")

    return {
        "consistent": bool(gated) and not diverged and not unverified,
        "diverged": diverged,
        "unverified": unverified,
    }
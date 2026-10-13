#!/usr/bin/env bash
#
# verify-onboarding.sh — the acceptance pass, end to end, on a machine that has
# never seen this project.
#
# The standing acceptance criterion is: clone the repo on a *different* computer,
# configure an account and a password, and immediately use it to operate the
# JumpServer-managed cluster for further Torch-FL development. Reading the code
# cannot confirm that; running it can. This script walks the whole first-run path in
# one go and stops at the first step that does not work, printing that step's real
# output:
#
#   init -> login -> list assets -> build hosts.yaml -> doctor (the gate)
#        -> jms_dev_discover -> one real jms_dev_run
#
# It never reads or writes the real ~/.config/jms-bridge: the config, the inventory
# and the credentials file are all pinned inside a fresh mktemp directory, and the
# run is forced onto keyring's null backend so a saved password lands in the scratch
# file rather than the real OS keychain. It deliberately does *not* move $HOME —
# ssh still needs the real ~/.ssh (keys, known_hosts) for the data plane.
#
# Usage:
#   scripts/verify-onboarding.sh
#
# Everything can be answered from the environment instead of a prompt:
#
#   JUMP_URL=https://jump.example.com     JumpServer base URL
#   SSH_PORT=2222                         KoKo SSH gateway port
#   LOGIN_USER=your-login                 the `login` dimension of login#account#asset
#   JMS_BRIDGE_PASSWORD=...               control-plane password (else prompted)
#   JMS_BRIDGE_SSH_PASSWORD=...           data-plane password, if the gateway needs one
#   DEFAULT_ACCOUNT=root                  system user on each asset (the `#account` field)
#   PICK="0,2"                            which assets to import (indices/names/id);
#                                         default: every asset, no prompt
#   HOSTS="fl-ascend=0:ascend-910c,fl-4090=1"   per-host override: name=ref[:platform[:account]]
#   CONTAINER=lvyufeng-torch-fl           container that holds the checkout
#   REPO_MATCH=torch                      substring a checkout path must contain
#   SOURCE=git                            git | proxy-git | push
#   PROXY=...                             required when SOURCE=proxy-git
#   MOUNT=... LOCAL=...                   required when SOURCE=push
#   REF=main                              the git ref jms_dev_run delivers
#
# Exit status is 0 only when every step — including a real cross-chip jms_dev_run —
# succeeded.
#
# Runs on the macOS system bash (3.2) as well as bash 5: no `printf -v`, no `local -n`.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCP_CALL="$REPO/scripts/mcp_call.py"
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/jmsb-verify.XXXXXX")"

if [ -t 1 ]; then B=$'\033[1m'; R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; N=$'\033[0m'
else B=; R=; G=; Y=; N=; fi
TAB="$(printf '\t')"
say()  { printf '\n%s== %s ==%s\n' "$B" "$*" "$N"; }
ok()   { printf '  %sOK%s   %s\n' "$G" "$N" "$*"; }
warn() { printf '  %sWARN%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '\n%sFAILED: %s%s\n' "$R" "$*" "$N" >&2; exit 1; }

LAST_OUT=""
step_failed() {
  printf '\n%sFAILED: %s%s\n' "$R" "$1" "$N" >&2
  printf '\n--- output of that step ---\n%s\n--- end ---\n' "${LAST_OUT:-<none>}" >&2
  SCRATCH_KEPT=1
  printf '\nscratch dir kept for inspection:\n  %s\n' "$SCRATCH" >&2
  exit 1
}

# Keep the scratch dir on failure, remove it on success.
trap 'st=$?; if [ "$st" = 0 ]; then rm -rf "$SCRATCH"; elif [ "${SCRATCH_KEPT:-0}" = 0 ]; then printf "\n(scratch kept for inspection: %s)\n" "$SCRATCH" >&2; fi' EXIT

# run "what" cmd...   -> stream the output; on a nonzero exit, show it and stop.
run() {
  local what="$1"; shift
  if ! LAST_OUT="$("$@" 2>&1)"; then step_failed "$what"; fi
  printf '%s\n' "$LAST_OUT"
}

# capture cmd...      -> record output+status without failing (for tools whose
#                        ok:false result we want to parse, not just report).
capture() { set +e; LAST_OUT="$("$@" 2>&1)"; LAST_STATUS=$?; set -e; }

prompt() { # prompt VAR "question" [default]
  local __var="$1" __q="$2" __def="${3-}" __ans=""
  [ -n "${!__var:-}" ] && return 0
  [ -t 0 ] || die "$__var is not set and stdin is not a terminal"
  if [ -n "$__def" ]; then
    read -r -p "$__q [$__def]: " __ans || true; __ans="${__ans:-$__def}"
  else
    read -r -p "$__q: " __ans || true
  fi
  # Not `printf -v`: that is a bash 4 builtin and macOS ships bash 3.2.
  eval "$__var=\$__ans"
}

# --- 0/9: locate a usable interpreter ----------------------------------------------

say "0/9  interpreter"
PY="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] && [ -x "$PY" ] || die "no python found; create .venv (python3 -m venv .venv && .venv/bin/pip install -e '.[dev]') or set PYTHON"
export PYTHONPATH="$REPO/src"
"$PY" -c 'import jms_bridge, mcp' 2>/dev/null || die "$PY cannot import jms_bridge+mcp; install the project into it (pip install -e .)"
ok "python  $PY"
ok "repo    $REPO"
ok "scratch $SCRATCH"

# --- 1/9: collect the deployment details -------------------------------------------

say "1/9  deployment details"
prompt JUMP_URL  "JumpServer base URL (https://...)"
prompt SSH_PORT  "KoKo SSH gateway port" "2222"
prompt LOGIN_USER "JumpServer login (the 'login' dimension of login#account#asset)"
prompt DEFAULT_ACCOUNT "system user on each asset" "root"
[ -n "$JUMP_URL" ] && [ -n "$LOGIN_USER" ] || die "JUMP_URL and LOGIN_USER must be non-empty"

if [ -z "${JMS_BRIDGE_PASSWORD:-}" ]; then
  [ -t 0 ] || die "JMS_BRIDGE_PASSWORD is not set and stdin is not a terminal"
  read -r -s -p "JumpServer password: " JMS_BRIDGE_PASSWORD; echo
  export JMS_BRIDGE_PASSWORD
fi
ok "url     $JUMP_URL"
ok "login   $LOGIN_USER  (account $DEFAULT_ACCOUNT)"

# Pin every bridge path inside the scratch dir, and force the null keyring so a
# saved password never reaches the real OS keychain.
export JMS_BRIDGE_CONFIG="$SCRATCH/config.toml"
export JMS_BRIDGE_INVENTORY="$SCRATCH/hosts.yaml"
export JMS_BRIDGE_CREDENTIALS="$SCRATCH/credentials.toml"
export PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring"

# The CLI, addressed through the package rather than the installed console script, so a
# checkout that has not been `pip install`-ed still runs — which is the state a fresh
# clone is in. `src/jms_bridge/__main__.py` is what makes `python -m` work.
jms() { PYTHONPATH="$REPO/src" "$PY" -m jms_bridge "$@"; }

SUPPORT="$REPO/scripts/verify_support.py"
mcp() { # mcp TOOL [ARGS_JSON] [mcp_call flags...]
  local tool="$1"; shift
  local args="{}"
  if [ "$#" -gt 0 ] && [ "${1#--}" = "$1" ]; then args="$1"; shift; fi
  "$PY" "$MCP_CALL" "$tool" "$args" "$@"
}

# --- 2/9: init, then point the scratch config at the real bastion -------------------

say "2/9  jms-bridge init -> $SCRATCH"
run "jms-bridge init" jms init

# `init` wrote the documented template; edit it in place, the way a first-run user
# does, so the pass exercises the template rather than a config this script invented.
# json.dumps gives a correctly quoted TOML string for whatever was typed.
"$PY" - "$JMS_BRIDGE_CONFIG" "$JUMP_URL" "$SSH_PORT" "$LOGIN_USER" <<'PYEOF'
import json, re, sys

path, url, port, login = sys.argv[1:5]
text = open(path).read()
for key, value in (("url", url), ("ssh_port", int(port)), ("login_user", login)):
    new, n = re.subn(
        rf"(?m)^{key}\s*=.*$",
        f"{key} = {json.dumps(value) if isinstance(value, str) else value}",
        text,
    )
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one '{key}' line, found {n}")
    text = new
open(path, "w").write(text)
PYEOF
grep -q "url = " "$JMS_BRIDGE_CONFIG" || die "config.toml lost its url line"
ok "config.toml edited ($JUMP_URL, ssh port $SSH_PORT, login $LOGIN_USER)"

# --- 3/9: log in once, and persist the password in the scratch credentials file -----

say "3/9  jms-bridge login --save-password"
run "jms-bridge login" jms login --save-password
grep -q 'session_token' "$JMS_BRIDGE_CONFIG" || die "login did not cache a session token in $JMS_BRIDGE_CONFIG"
mode() { stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1" 2>/dev/null || echo '?'; }
ok "session token cached (config mode $(mode "$JMS_BRIDGE_CONFIG"), credentials $(mode "$JMS_BRIDGE_CREDENTIALS"))"

# --- 4/9: the asset ids the inventory needs, over the real MCP protocol -------------

say "4/9  jms_list_assets  (over MCP stdio)"
capture mcp jms_list_assets
[ "$LAST_STATUS" -eq 0 ] || step_failed "jms_list_assets (control plane)"
printf '%s' "$LAST_OUT" > "$SCRATCH/assets.json"
capture "$PY" "$SUPPORT" assets --file "$SCRATCH/assets.json"
[ "$LAST_STATUS" -eq 0 ] || step_failed "read the jms_list_assets result"
printf '%s\n' "$LAST_OUT" | while IFS="$TAB" read -r idx name addr plat id; do
  case "$idx" in warning:*) printf '  %s\n' "$idx" >&2; continue ;; esac
  printf '  [%s] %-24s %-16s %s\n' "$idx" "$name" "$addr" "$plat"
done
[ -n "$LAST_OUT" ] || step_failed "jms_list_assets returned no assets (no account bound?)"

# --- 5/9: turn those assets into an inventory ---------------------------------------

say "5/9  hosts.yaml"
if [ -z "${HOSTS-}${PICK-}" ] && [ -t 0 ]; then
  read -r -p "Import which assets? (indices/names comma-separated, empty = all): " PICK || true
fi
capture "$PY" "$SUPPORT" mkhosts \
  --assets "$SCRATCH/assets.json" --out "$JMS_BRIDGE_INVENTORY" \
  --login "$LOGIN_USER" --account "$DEFAULT_ACCOUNT" \
  --pick "${PICK-}" --hosts "${HOSTS-}"
[ "$LAST_STATUS" -eq 0 ] || step_failed "build hosts.yaml from jms_list_assets"
printf '%s\n' "$LAST_OUT" | while IFS="$TAB" read -r name plat id; do
  case "$name" in warning:*) printf '  %s\n' "$name" >&2; continue ;; esac
  printf '  %-24s platform=%-12s id=%s\n' "$name" "$plat" "$id"
done
ok "inventory written to $JMS_BRIDGE_INVENTORY"

# --- 6/9: doctor is the gate --------------------------------------------------------

say "6/9  jms-bridge doctor (the gate)"
run "jms-bridge doctor" jms doctor all
ok "tooling, config, control plane and data plane all green"

# --- 7/9: discover the checkouts, and append the workspaces block -------------------

say "7/9  jms_dev_discover  (builds the workspaces: block)"
CONTAINER="${CONTAINER:-lvyufeng-torch-fl}"
SOURCE="${SOURCE:-git}"
REPO_MATCH="${REPO_MATCH:-torch}"
DISC_ARGS="$("$PY" -c 'import json,sys; print(json.dumps({"targets":["all"],"container":sys.argv[1],"source":sys.argv[2],"repo_match":sys.argv[3],"proxy":sys.argv[4] or None,"mount":sys.argv[5] or None,"local":sys.argv[6] or None}))' \
  "$CONTAINER" "$SOURCE" "$REPO_MATCH" "${PROXY-}" "${MOUNT-}" "${LOCAL-}")"
capture mcp jms_dev_discover "$DISC_ARGS" --timeout 300
[ "$LAST_STATUS" -ne 2 ] || step_failed "jms_dev_discover (the call never completed)"
printf '%s' "$LAST_OUT" > "$SCRATCH/discover.json"
capture "$PY" "$SUPPORT" discover --file "$SCRATCH/discover.json" \
  --hosts "$JMS_BRIDGE_INVENTORY" --append
printf '%s\n' "$LAST_OUT"
[ "$LAST_STATUS" -eq 0 ] || step_failed "jms_dev_discover (a host it could not inspect is in the output above)"
ok "workspaces block appended to $JMS_BRIDGE_INVENTORY"

# --- 8/9: the inventory still loads with the workspaces block -----------------------

say "8/9  inventory reload"
run "load the inventory" "$PY" -c '
import sys
from jms_bridge.inventory import Inventory
inv = Inventory.load()
print(f"{len(inv.hosts)} hosts, {len(inv.workspaces)} workspaces: {sorted(inv.workspaces)}")
'
ok "workspaces block loads"

# --- 9/9: one real cross-chip run ---------------------------------------------------

say "9/9  jms_dev_run  (the real thing)"
REF="${REF:-main}"
RUN_ARGS="$("$PY" -c 'import json,sys; print(json.dumps({"targets":["all"],"ref":sys.argv[1]}))' "$REF")"
capture mcp jms_dev_run "$RUN_ARGS" --timeout 3600
[ "$LAST_STATUS" -ne 2 ] || step_failed "jms_dev_run (the call never completed)"
printf '%s' "$LAST_OUT" > "$SCRATCH/devrun.json"
set +e
"$PY" "$SUPPORT" devrun --file "$SCRATCH/devrun.json"
RUN_RC=$?
set -e

if [ "$RUN_RC" -eq 0 ]; then
  printf '\n%sACCEPTANCE PASSED%s — cloned, configured, and drove the fleet end to end.\n' "$G" "$N"
else
  printf '\n%sACCEPTANCE FAILED%s — the table above shows why; the run kept its scratch dir.\n' "$R" "$N"
  SCRATCH_KEPT=1
fi
exit "$RUN_RC"

"""Credential persistence for jms-agent-bridge.

A JumpServer *password* is the one secret a user may want stored: it turns the
documented first-run step (``export JMS_BRIDGE_PASSWORD=... && jms-bridge login``)
into one that survives a token expiry, a new shell, or a fresh clone — without the
password ever landing in the hand-written, shareable ``config.toml``.

Storage is layered, most-preferred first::

    JMS_BRIDGE_PASSWORD env  >  OS keyring  >  ~/.config/jms-bridge/credentials.toml

* The environment always wins, so CI and throwaway shells never read or write disk.
* The OS keyring (macOS Keychain, Freedesktop Secret Service, Windows Credential
  Locker) is used when the optional ``keyring`` extra is installed *and* a backend
  actually works; it keeps the password out of a dotfile and out of home-directory
  backups. A headless host with no unlocked keyring degrades quietly to the file.
* The file is the always-available fallback: machine-written, mode 600, and
  deletable with ``rm`` to revoke. A password is only ever written there when the
  user asked for it (``login --save-password``) *and* no keyring could take it.

Only the password lives here. The session token minted by ``login`` is a different,
ephemeral secret and stays in ``config.toml`` (also mode 600) — see
:mod:`jms_bridge.config`.
"""

from __future__ import annotations

import json
import os

from .config import credentials_path, read_toml

#: Keyring service name. The account is a fixed constant because one config targets
#: one bastion — multi-bastion is intentionally out of scope, so a per-bastion key
#: would be a speculative dimension.
_SERVICE = "jms-agent-bridge"
_ACCOUNT = "jumpserver"

#: Shown next to the "keyring unavailable" state so a user knows the remedy.
KEYRING_HINT = "install the 'keyring' extra"


def _keyring():
    """Return the ``keyring`` module if it is installed *and* has a usable backend.

    Imported lazily and defensively: ``keyring`` is an optional extra, and on a
    headless host a backend can import yet be unable to reach a store (no D-Bus, a
    locked keychain). Both are reasons to fall back to the file, not to raise at the
    call site, so every failure here returns ``None``.
    """
    try:
        import keyring
    except Exception:  # ImportError, or a backend module that fails to import
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    # keyring's contract: backends that cannot store anything (the "fail" backend,
    # used when no platform store is found) report priority <= 0.
    if getattr(backend, "priority", 0) <= 0:
        return None
    return keyring


def keyring_backend() -> str | None:
    """Human name of the active keyring backend, or ``None`` if keyring is unusable."""
    keyring = _keyring()
    if keyring is None:
        return None
    backend = keyring.get_keyring()
    module = type(backend).__module__
    return f"{module}.{type(backend).__name__}"


def _read_file_password() -> str | None:
    try:
        data = read_toml(credentials_path())
    except Exception:  # a hand-broken credentials file must not be fatal
        return None
    password = data.get("credentials", {}).get("password")
    return password or None


def _write_file(password: str) -> os.PathLike:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "# Machine-written by `jms-bridge login --save-password`. Plain text, mode 600.\n"
        "# Delete this file (or run `jms-bridge login --forget`) to revoke the password.\n"
        "[credentials]\n"
        f"password = {json.dumps(password)}\n"  # JSON string escaping == TOML basic string
    )
    # Open with 600 so the secret is never even briefly world-readable, and re-chmod in
    # case the file pre-existed with looser bits (a truncated O_CREAT keeps them).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(body)
    os.chmod(path, 0o600)
    return path


def _remove_file() -> bool:
    try:
        credentials_path().unlink()
    except OSError:
        return False
    return True


def get_password() -> str | None:
    """Resolve the bastion password: env, then keyring, then the credentials file."""
    env = os.environ.get("JMS_BRIDGE_PASSWORD")
    if env:
        return env
    keyring = _keyring()
    if keyring is not None:
        try:
            stored = keyring.get_password(_SERVICE, _ACCOUNT)
        except Exception:
            stored = None
        if stored:
            return stored
    return _read_file_password()


def save_password(password: str) -> str:
    """Persist *password* where it is safest; return a human description of where.

    Prefers the keyring, and removes any plaintext copy when that succeeds, so the
    password never exists in two stores at once. Falls back to the mode-600 file
    when no keyring backend is available.
    """
    if not password:
        raise ValueError("refusing to store an empty password")
    keyring = _keyring()
    if keyring is not None:
        try:
            keyring.set_password(_SERVICE, _ACCOUNT, password)
        except Exception:
            pass
        else:
            _remove_file()
            return f"keyring ({keyring_backend()})"
    path = _write_file(password)
    return f"file {path} (mode 600; {KEYRING_HINT} to keep it out of plain text)"


def forget_password() -> list[str]:
    """Delete every stored copy of the password; return descriptions of what was cleared."""
    cleared: list[str] = []
    keyring = _keyring()
    if keyring is not None:
        try:
            keyring.delete_password(_SERVICE, _ACCOUNT)
        except Exception:
            pass  # nothing stored, or no backend; the file may still hold one
        else:
            cleared.append(f"keyring ({keyring_backend()})")
    if _remove_file():
        cleared.append("credentials file")
    return cleared


def stored_password_source() -> str | None:
    """Where a saved password lives (for ``doctor``); ``None`` if none is stored."""
    keyring = _keyring()
    if keyring is not None:
        try:
            if keyring.get_password(_SERVICE, _ACCOUNT):
                return f"keyring ({keyring_backend()})"
        except Exception:
            pass
    if _read_file_password() is not None:
        return f"file {credentials_path()}"
    return None

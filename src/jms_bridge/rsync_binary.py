"""Resolve and validate the local rsync without blocking the resident event loop."""

import asyncio
import os
import re
import shutil
import subprocess

from .async_io import offload
from .config import ConfigError

PROBE_TIMEOUT = 3
_INSTALL_HINT = "Install rsync 3.x (on macOS: brew install rsync) and put it first on PATH."
_VERSION = re.compile(r"^\s*rsync\s+version\s+(\d+)\.(\d+)\b", re.IGNORECASE | re.MULTILINE)
_UNSUPPORTED_OPTION = re.compile(
    r"(?:invalid|illegal|unknown|unrecognized|unsupported) option[^\n]*"
    r"(?:--protect-args|--secluded-args|(?<!\w)-s\b|--\s*['\"`]?s\b)",
    re.IGNORECASE,
)


def _identity(binary):
    """Notice upgrades/replacements even when PATH still names the same file."""
    st = os.stat(binary)
    return binary, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


class RsyncBinary:
    """One bounded, successful capability cache per resident SSH runner.

    Only a successful probe is cached. Failures may be transient, and an installed
    replacement or a changed PATH must be usable without restarting the service.
    The lock coalesces concurrent cold lookups; offload also waits for a cancelled
    caller's bounded probe to finish before releasing that lock.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._validated = None

    async def resolve(self):
        async with self._lock:
            # PATH lookup and stat may also touch slow storage. Keep all of the
            # local preflight off-loop, before SSH or an execution slot is used.
            return await offload(self._resolve)

    def _resolve(self):
        binary = shutil.which("rsync")
        if binary is None:
            raise ConfigError(
                f"Local rsync is not installed or not executable on PATH. {_INSTALL_HINT}"
            )
        # Pin the resolved executable, not another PATH lookup after SSH/queue wait.
        binary = os.path.realpath(binary)
        try:
            identity = _identity(binary)
        except OSError as exc:
            raise ConfigError(
                f"Cannot inspect local rsync at {binary} ({type(exc).__name__}); "
                "check the installation and permissions. Transfer not started."
            ) from None
        if identity == self._validated:
            return binary
        try:
            probe = subprocess.run(
                [binary, "-s", "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=PROBE_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ConfigError(
                f"Local rsync capability check timed out after {PROBE_TIMEOUT}s at {binary}; "
                "check the installation and retry. Transfer not started."
            ) from None
        except OSError as exc:
            raise ConfigError(
                f"Cannot run local rsync capability check at {binary} "
                f"({type(exc).__name__}); check the installation and permissions. "
                "Transfer not started."
            ) from None
        output = probe.stdout + "\n" + probe.stderr
        version = _VERSION.search(output)
        if "openrsync" in output.lower() or _UNSUPPORTED_OPTION.search(output):
            raise ConfigError(
                f"Local rsync at {binary} does not support --protect-args (-s), "
                f"required by compact transfers. {_INSTALL_HINT} Transfer not started."
            )
        if probe.returncode != 0:
            raise ConfigError(
                f"Local rsync capability check (-s --version) failed at {binary} "
                f"with exit code {probe.returncode}; support could not be verified. "
                "Check the installation. Transfer not started."
            )
        if version is None:
            raise ConfigError(
                f"Local rsync capability check at {binary} returned unrecognized version output; "
                "support could not be verified. Transfer not started."
            )
        if int(version[1]) < 3:
            raise ConfigError(
                f"Local rsync at {binary} is older than 3.0 and lacks --protect-args (-s). "
                f"{_INSTALL_HINT} Transfer not started."
            )
        try:
            unchanged = _identity(binary) == identity
        except OSError:
            unchanged = False
        if not unchanged:
            raise ConfigError(
                "Local rsync changed during its capability check; retry before transferring. "
                "Transfer not started."
            )
        self._validated = identity
        return binary

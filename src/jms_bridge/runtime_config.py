"""Optional compact-service settings; deployment addresses are supplied by the user."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .config import BastionConfig, ConfigError, config_path, read_toml


def default_state_dir() -> Path:
    if value := os.environ.get("JMS_BRIDGE_STATE_DIR"):
        return Path(value).expanduser()
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    return base / "jms-bridge"


@dataclass
class BridgeSettings:
    state_dir: Path = field(default_factory=default_state_dir)
    refresh_interval: int = 3600
    execution_limit: int | None = None
    inspection_reserved: int = 0
    queue_timeout: float = 60
    queue_limit: int = 64
    inspection_queue_limit: int = 16
    result_ttl_seconds: float = 604800
    result_cleanup_interval_seconds: float = 3600
    preferred_account: str | None = None
    rest_proxy: str | None = None
    ssh_host: str | None = None
    ssh_proxy_command: list[str] = field(default_factory=list)
    known_hosts: Path | None = None
    host_key_alias: str | None = None

    def gateway_host(self, cfg: BastionConfig) -> str:
        host = self.ssh_host or urlparse(cfg.url).hostname
        if not host:
            raise ConfigError("configure bridge.ssh_host or a valid bastion.url")
        return host


def load_settings() -> BridgeSettings:
    data = read_toml(config_path()).get("bridge", {})
    if not isinstance(data, dict):
        raise ConfigError("[bridge] must be a table")
    interval = data.get("refresh_interval", 3600)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ConfigError("bridge.refresh_interval must be a positive integer")
    import math

    for name in ("execution_limit", "inspection_reserved"):
        value = data.get(name)
        if value is not None and (
            type(value) is not int or value < (0 if name == "inspection_reserved" else 1)
        ):
            raise ConfigError(f"bridge.{name} must be a valid integer")
    for name in ("queue_limit", "inspection_queue_limit"):
        value = data.get(name, 0)
        if type(value) is not int or value < 0:
            raise ConfigError(f"bridge.{name} must be a nonnegative integer")
    for name in ("queue_timeout", "result_ttl_seconds", "result_cleanup_interval_seconds"):
        value = data.get(name, 1)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ConfigError(f"bridge.{name} must be positive and finite")
    if (
        data.get("execution_limit") is not None
        and data.get("inspection_reserved", 0) >= data["execution_limit"]
    ):
        raise ConfigError("bridge.inspection_reserved must be less than execution_limit")
    command = data.get("ssh_proxy_command", [])
    if not isinstance(command, list) or any(
        not isinstance(s, str) or not s or "\x00" in s for s in command
    ):
        raise ConfigError("bridge.ssh_proxy_command must be an argv list of nonempty strings")
    for name in (
        "state_dir",
        "preferred_account",
        "rest_proxy",
        "ssh_host",
        "known_hosts",
        "host_key_alias",
    ):
        if name in data and (not isinstance(data[name], str) or not data[name]):
            raise ConfigError(f"bridge.{name} must be a nonempty string")
    state = os.environ.get("JMS_BRIDGE_STATE_DIR") or data.get("state_dir")
    return BridgeSettings(
        state_dir=Path(state).expanduser() if state else default_state_dir(),
        refresh_interval=interval,
        **{
            name: data[name]
            for name in (
                "execution_limit",
                "inspection_reserved",
                "queue_timeout",
                "queue_limit",
                "inspection_queue_limit",
                "result_ttl_seconds",
                "result_cleanup_interval_seconds",
            )
            if name in data
        },
        preferred_account=data.get("preferred_account"),
        rest_proxy=data.get("rest_proxy"),
        ssh_host=data.get("ssh_host"),
        ssh_proxy_command=command,
        known_hosts=Path(data["known_hosts"]).expanduser() if data.get("known_hosts") else None,
        host_key_alias=data.get("host_key_alias"),
    )


class StateLock:
    """One resident process owns a state directory and its persistent-shell registry."""

    def __init__(self, directory):
        self.path = directory / "service.lock"
        self.fd = None

    def __enter__(self):
        import fcntl

        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            self.fd = None
            raise ConfigError(
                "state directory is already in use; connect to the resident HTTP service "
                "or configure a separate bridge.state_dir"
            ) from None
        return self

    def __exit__(self, *args):
        os.close(self.fd)
        self.fd = None

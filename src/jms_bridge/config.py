"""Configuration loading for jms-agent-bridge.

Config lives at ``~/.config/jms-bridge/config.toml`` (overridable with the
``JMS_BRIDGE_CONFIG`` env var). Secrets are never required to be in this file:
every value can be supplied or overridden from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib  # type: ignore

DEFAULT_CONFIG_PATH = "~/.config/jms-bridge/config.toml"
DEFAULT_INVENTORY_PATH = "~/.config/jms-bridge/hosts.yaml"


@dataclass
class BastionConfig:
    """Connection settings for one JumpServer deployment."""

    #: Web/API base URL, e.g. ``https://bastion.example.com``
    url: str
    #: KoKo SSH gateway port (JumpServer's default is 2222; some deployments use 2224)
    ssh_port: int = 2222
    #: SSH user on the bastion for the *login* dimension, used in ``login#account#asset``
    login_user: str | None = None
    #: Path to the SSH private key used for the data plane. None -> ssh-agent / default.
    ssh_key: Path | None = None
    #: Verify TLS certificates. Turn off only for self-signed internal certs.
    verify_tls: bool = True
    #: Request timeout (seconds) for the REST control plane.
    api_timeout: float = 30.0
    #: Set if the deployment enforces multi-org; sent as the ``X-JMS-ORG`` header.
    org_id: str | None = None

    # --- REST credentials (control plane) ---
    #: JumpServer "access key" id (个人设置 > 访问密钥).
    access_key_id: str | None = None
    #: JumpServer "access key" secret.
    access_key_secret: str | None = None
    #: Private token, if the deployment uses that instead of an access key.
    private_token: str | None = None
    #: A session token from ``jms-bridge login`` (sent as ``Bearer``, unlike a
    #: private token which is sent as ``Token``). Cached in the config file.
    session_token: str | None = None
    #: Explicit base path for the REST API. JumpServer is usually ``/api/v1``.
    api_base: str = "/api/v1"

    def api_url(self, path: str) -> str:
        return f"{self.url.rstrip('/')}{self.api_base}/{path.lstrip('/')}"


def _expand(value: str | None) -> str | None:
    return os.path.expanduser(value) if isinstance(value, str) else value


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None


def load_config(path: str | Path | None = None) -> BastionConfig:
    """Load config from TOML, then let environment variables override.

    Environment overrides (useful for CI and for keeping secrets out of the file)::

        JMS_BRIDGE_URL, JMS_BRIDGE_SSH_PORT, JMS_BRIDGE_LOGIN_USER,
        JMS_BRIDGE_SSH_KEY, JMS_BRIDGE_ACCESS_KEY_ID, JMS_BRIDGE_ACCESS_KEY_SECRET,
        JMS_BRIDGE_PRIVATE_TOKEN, JMS_BRIDGE_ORG_ID, JMS_BRIDGE_VERIFY_TLS
    """

    cfg_path = Path(_expand(str(path)) if path else _expand(str(DEFAULT_CONFIG_PATH)))
    if cfg_path_or_env := _env("JMS_BRIDGE_CONFIG"):
        cfg_path = Path(cfg_path_or_env).expanduser()

    data: dict = {}
    if cfg_path.exists():
        with cfg_path.open("rb") as fh:
            data = tomllib.load(fh)

    bastion = data.get("bastion", {})

    url = _env("JMS_BRIDGE_URL") or bastion.get("url")
    if not url:
        raise ConfigError(
            f"No bastion URL configured. Set 'bastion.url' in {cfg_path} "
            "or the JMS_BRIDGE_URL environment variable."
        )

    verify_tls_env = _env("JMS_BRIDGE_VERIFY_TLS")
    verify_tls = (
        verify_tls_env.lower() not in {"0", "false", "no"}
        if verify_tls_env
        else bastion.get("verify_tls", True)
    )

    ssh_key = _env("JMS_BRIDGE_SSH_KEY") or bastion.get("ssh_key")

    return BastionConfig(
        url=url.rstrip("/"),
        ssh_port=int(_env("JMS_BRIDGE_SSH_PORT") or bastion.get("ssh_port", 2222)),
        login_user=_env("JMS_BRIDGE_LOGIN_USER") or bastion.get("login_user"),
        ssh_key=Path(_expand(ssh_key)).expanduser() if ssh_key else None,
        verify_tls=bool(verify_tls),
        api_timeout=float(bastion.get("api_timeout", 30.0)),
        org_id=_env("JMS_BRIDGE_ORG_ID") or bastion.get("org_id"),
        access_key_id=_env("JMS_BRIDGE_ACCESS_KEY_ID") or bastion.get("access_key_id"),
        access_key_secret=_env("JMS_BRIDGE_ACCESS_KEY_SECRET") or bastion.get("access_key_secret"),
        private_token=_env("JMS_BRIDGE_PRIVATE_TOKEN") or bastion.get("private_token"),
        session_token=_env("JMS_BRIDGE_SESSION_TOKEN") or bastion.get("session_token"),
        api_base=bastion.get("api_base", "/api/v1"),
    )


def inventory_path(path: str | Path | None = None) -> Path:
    if env := _env("JMS_BRIDGE_INVENTORY"):
        return Path(env).expanduser()
    if path:
        return Path(_expand(str(path))).expanduser()
    return Path(_expand(str(DEFAULT_INVENTORY_PATH)))


class ConfigError(RuntimeError):
    """Raised when configuration is missing or unusable."""
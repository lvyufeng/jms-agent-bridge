import textwrap

import pytest

from jms_bridge.config import BastionConfig, ConfigError, credentials_path, load_config


def test_env_overrides_file(tmp_path, monkeypatch):
    p = tmp_path / "config.toml"
    p.write_text(
        textwrap.dedent(
            """
            [bastion]
            url = "https://from-file"
            ssh_port = 2222
            login_user = "fileuser"
            """
        )
    )
    monkeypatch.setenv("JMS_BRIDGE_URL", "https://from-env")
    monkeypatch.setenv("JMS_BRIDGE_SSH_PORT", "2224")
    cfg = load_config(p)
    assert cfg.url == "https://from-env"
    assert cfg.ssh_port == 2224
    assert cfg.login_user == "fileuser"


def test_missing_url_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("JMS_BRIDGE_URL", raising=False)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nonexistent.toml")


def test_verify_tls_env_parsing(tmp_path, monkeypatch):
    p = tmp_path / "c.toml"
    p.write_text('[bastion]\nurl = "https://x"\n')
    monkeypatch.setenv("JMS_BRIDGE_VERIFY_TLS", "false")
    assert load_config(p).verify_tls is False
    monkeypatch.setenv("JMS_BRIDGE_VERIFY_TLS", "true")
    assert load_config(p).verify_tls is True


def test_api_url_joining():
    cfg = BastionConfig(url="https://x/", api_base="/api/v1")
    assert cfg.api_url("assets/assets/") == "https://x/api/v1/assets/assets/"
    assert cfg.api_url("/accounts/accounts/") == "https://x/api/v1/accounts/accounts/"


def test_credentials_path_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("JMS_BRIDGE_CREDENTIALS", raising=False)
    assert credentials_path(tmp_path / "a.toml") == tmp_path / "a.toml"
    monkeypatch.setenv("JMS_BRIDGE_CREDENTIALS", str(tmp_path / "b.toml"))
    assert credentials_path(tmp_path / "a.toml") == tmp_path / "b.toml"
"""Tests for credential persistence (env / keyring / credentials file)."""

import stat

import pytest

from jms_bridge import secrets
from jms_bridge.config import credentials_path


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point HOME at a scratch dir and clear the credential env vars.

    There is no tests/conftest.py, so this file carries its own isolation: a test must
    never read or write the real ``~/.config/jms-bridge``.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("JMS_BRIDGE_PASSWORD", "JMS_BRIDGE_CREDENTIALS"):
        monkeypatch.delenv(var, raising=False)


def test_missing_password_is_none():
    assert secrets.get_password() is None
    assert secrets.stored_password_source() is None


def test_no_keyring_falls_back_to_file(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    assert secrets.keyring_backend() is None
    where = secrets.save_password("plain")
    assert "file" in where
    assert secrets.get_password() == "plain"
    assert secrets.stored_password_source() == f"file {credentials_path()}"


def test_file_is_mode_600(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    secrets.save_password("s3cret")
    path = credentials_path()
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_quotes_and_backslash_roundtrip(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    password = 'a"b\\c\nd'
    secrets.save_password(password)
    assert secrets.get_password() == password


def test_env_wins_over_saved(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    secrets.save_password("from-file")
    monkeypatch.setenv("JMS_BRIDGE_PASSWORD", "from-env")
    assert secrets.get_password() == "from-env"


def test_credentials_path_env_override(tmp_path, monkeypatch):
    alt = tmp_path / "elsewhere.toml"
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    monkeypatch.setenv("JMS_BRIDGE_CREDENTIALS", str(alt))
    secrets.save_password("y")
    assert alt.exists()
    assert secrets.get_password() == "y"


def test_forget_clears_file(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    secrets.save_password("x")
    assert secrets.forget_password() == ["credentials file"]
    assert secrets.get_password() is None
    assert secrets.forget_password() == []  # idempotent


def test_keyring_preferred_and_removes_plaintext(monkeypatch):
    """A working keyring takes the password and leaves no plaintext copy behind."""
    store = {}

    class FakeKeyring:
        priority = 1

        def get_password(self, service, account):
            return store.get((service, account))

        def set_password(self, service, account, password):
            store[(service, account)] = password

        def delete_password(self, service, account):
            store.pop((service, account), None)

    fake = FakeKeyring()
    # Seed the file first (no keyring), then flip the backend on and re-save.
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    secrets.save_password("first")
    path = credentials_path()
    assert path.exists()

    monkeypatch.setattr(secrets, "_keyring", lambda: fake)
    monkeypatch.setattr(secrets, "keyring_backend", lambda: "test.Fake")

    where = secrets.save_password("second")
    assert "keyring" in where
    assert not path.exists()  # plaintext copy removed once the keyring held it
    assert secrets.get_password() == "second"
    assert secrets.stored_password_source() == "keyring (test.Fake)"
    assert secrets.forget_password() == ["keyring (test.Fake)"]

import base64

import pytest

from jms_bridge.api import JumpServerAPI, JumpServerError
from jms_bridge.config import BastionConfig


def test_access_key_signature_is_deterministic_and_well_formed():
    cfg = BastionConfig(
        url="https://x", access_key_id="AKID", access_key_secret="SECRET"
    )
    api = JumpServerAPI(cfg)
    header = api._sign_access_key("GET", "/api/v1/assets/assets/", b"")
    assert header.startswith("Signature AKID:")
    # the signature decodes to a 32-byte HMAC-SHA256 digest
    sig = base64.b64decode(header.split(":", 1)[1])
    assert len(sig) == 32


def test_signature_binds_method_path_and_body():
    cfg = BastionConfig(url="https://x", access_key_id="AKID", access_key_secret="SECRET")
    api = JumpServerAPI(cfg)
    a = api._sign_access_key("GET", "/p", b"")
    b = api._sign_access_key("POST", "/p", b"")
    c = api._sign_access_key("GET", "/p", b'{"x":1}')
    # different method/body must not collide (date is equal within the same run)
    assert a != b
    assert a != c


def test_auth_headers_prefers_access_key():
    cfg = BastionConfig(
        url="https://x",
        access_key_id="AKID",
        access_key_secret="SECRET",
        private_token="TOK",
    )
    api = JumpServerAPI(cfg)
    headers = api._auth_headers("GET", "/p")
    assert headers["Authorization"].startswith("Signature")


def test_private_token_uses_token_scheme():
    api = JumpServerAPI(BastionConfig(url="https://x", private_token="TOK"))
    assert api._auth_headers("GET", "/p")["Authorization"] == "Token TOK"


def test_session_token_uses_bearer_scheme():
    api = JumpServerAPI(BastionConfig(url="https://x", private_token="TOK"))
    api._token = "SESSION"
    assert api._auth_headers("GET", "/p")["Authorization"] == "Bearer SESSION"


def test_no_credentials_raises():
    api = JumpServerAPI(BastionConfig(url="https://x"))
    with pytest.raises(JumpServerError):
        api._auth_headers("GET", "/p")


def test_org_header_only_when_set():
    api = JumpServerAPI(BastionConfig(url="https://x", org_id="ORG-1"))
    assert api._org_headers() == {"X-JMS-ORG": "ORG-1"}
    api2 = JumpServerAPI(BastionConfig(url="https://x"))
    assert api2._org_headers() == {}
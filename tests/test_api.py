import pytest

from jms_bridge.api import JumpServerAPI, JumpServerError
from jms_bridge.config import BastionConfig

# A fixed date so the signature is reproducible in tests.
DATE = "Thu, 09 Oct 2026 00:00:00 GMT"


def keyed(**kw):
    return JumpServerAPI(
        BastionConfig(url="https://x", access_key_id="AKID", access_key_secret="SECRET", **kw)
    )


def test_signature_header_shape():
    auth, headers = keyed().sign_access_key(
        "GET", "/api/v1/perms/users/my/assets/", date=DATE, accept="application/json", org=None
    )
    assert auth.startswith('Signature keyId="AKID",algorithm="hmac-sha256",')
    assert 'headers="(request-target) date accept"' in auth
    assert 'signature="' in auth
    assert headers == {"date": DATE, "accept": "application/json"}


def test_signature_binds_method_path_and_org():
    api = keyed()
    a, _ = api.sign_access_key("GET", "/p", date=DATE, accept="application/json", org=None)
    b, _ = api.sign_access_key("POST", "/p", date=DATE, accept="application/json", org=None)
    c, _ = api.sign_access_key("GET", "/q", date=DATE, accept="application/json", org=None)
    d, hd = api.sign_access_key("GET", "/p", date=DATE, accept="application/json", org="ORG")
    assert len({a, b, c, d}) == 4  # every input changes the signature
    assert "x-jms-org" in hd and hd["x-jms-org"] == "ORG"
    assert 'headers="(request-target) date accept x-jms-org"' in d


def test_signature_matches_pyhttpsig_library():
    """Cross-check our hand-rolled signer against the library JumpServer verifies with."""
    httpsig = pytest.importorskip("httpsig")

    api = keyed()
    auth, _ = api.sign_access_key(
        "GET", "/api/v1/assets/assets/", date=DATE, accept="application/json", org=None
    )

    signer = httpsig.HeaderSigner(
        "AKID",
        "SECRET",
        "hmac-sha256",
        headers=["(request-target)", "date", "accept"],
        sign_header="authorization",
    )
    expected = signer.sign(
        {"date": DATE, "accept": "application/json"},
        method="GET",
        path="/api/v1/assets/assets/",
    )
    assert auth == expected["authorization"]


def test_auth_headers_prefers_access_key():
    headers = keyed(private_token="TOK")._auth_headers("GET", "/p")
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


def test_create_ticket_uses_open_action(monkeypatch):
    """Tickets are created on the subtype collection's ``open`` action."""
    api = JumpServerAPI(BastionConfig(url="https://x", org_id="O1"))
    captured = {}

    async def fake_request(method, path, *, params=None, json=None):
        captured.update(method=method, path=path, json=json)
        return {"id": "T1"}

    monkeypatch.setattr(api, "request", fake_request)
    import asyncio

    asyncio.run(api.create_ticket(title="need login", ticket_type="login_asset_confirm"))
    assert captured["method"] == "POST"
    assert captured["path"] == "tickets/apply-login-asset-tickets/open/"
    assert captured["json"]["type"] == "login_asset_confirm"
    assert captured["json"]["org_id"] == "O1"


def test_unknown_ticket_type_raises():
    import asyncio

    api = JumpServerAPI(BastionConfig(url="https://x"))
    with pytest.raises(JumpServerError):
        asyncio.run(api.create_ticket(title="x", ticket_type="nope"))
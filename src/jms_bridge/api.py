"""JumpServer REST control plane.

Used for what the API is genuinely good at: discovering assets and their accounts,
reading the audit trail, and raising approval tickets. Anything that *executes* on
a host goes through the SSH data plane instead (see :mod:`jms_bridge.ssh`).

Authentication supports the two schemes JumpServer deployments actually use:

* **Private token** — ``Authorization: Token <token>``
* **Access key** — request signing with an id/secret pair

Session login (username/password → bearer token) is available via
:meth:`JumpServerAPI.login_with_password` but the bridge prefers a long-lived
token or access key so nothing asks for a password at runtime.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import format_datetime

import httpx

from .config import BastionConfig

log = logging.getLogger("jms_bridge.api")


class JumpServerError(RuntimeError):
    """Raised on a non-2xx response or an unusable API response."""


@dataclass
class JumpServerAPI:
    cfg: BastionConfig
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _token: str | None = field(default=None, init=False, repr=False)

    # --- transport -------------------------------------------------------------

    async def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.cfg.url,
                timeout=self.cfg.api_timeout,
                verify=self.cfg.verify_tls,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _org_headers(self) -> dict[str, str]:
        return {"X-JMS-ORG": self.cfg.org_id} if self.cfg.org_id else {}

    def _auth_headers(self, method: str, path: str, body: bytes = b"") -> dict[str, str]:
        """Build the auth header for one request, choosing the configured scheme."""
        if self.cfg.access_key_id and self.cfg.access_key_secret:
            return {"Authorization": self._sign_access_key(method, path, body)}
        token = self._token or self.cfg.private_token
        if token:
            # Private tokens use the "Token" scheme; session tokens use "Bearer".
            scheme = "Token" if (self.cfg.private_token and not self._token) else "Bearer"
            return {"Authorization": f"{scheme} {token}"}
        raise JumpServerError(
            "No JumpServer credentials configured. Set JMS_BRIDGE_ACCESS_KEY_ID/"
            "JMS_BRIDGE_ACCESS_KEY_SECRET or JMS_BRIDGE_PRIVATE_TOKEN, or call "
            "login_with_password()."
        )

    def _sign_access_key(self, method: str, path: str, body: bytes = b"") -> str:
        """Sign a request with the access-key id/secret (HMAC-SHA256).

        Header form: ``Authorization: Signature <access_key_id>:<base64 signature>``.
        The signed string binds method, path, body digest, content type and date, so
        a replayed or mutated request fails verification.
        """
        assert self.cfg.access_key_id and self.cfg.access_key_secret

        content_type = "application/json"
        date = format_datetime(datetime.now(timezone.utc))
        body_hash = base64.b64encode(hashlib.sha256(body).digest()).decode()
        string_to_sign = "\n".join(
            [method.upper(), path, content_type, date, body_hash]
        )
        signature = base64.b64encode(
            hmac.new(
                self.cfg.access_key_secret.encode(),
                string_to_sign.encode(),
                hashlib.sha256,
            ).digest()
        ).decode()
        return f"Signature {self.cfg.access_key_id}:{signature}"

    async def request(
        self, method: str, path: str, *, params: dict | None = None, json: dict | None = None
    ) -> dict:
        body = b"" if json is None else __import__("json").dumps(json).encode()
        # The signing path must include the query string as sent.
        signed_path = path
        if params:
            qs = str(httpx.QueryParams(params))
            signed_path = f"{path}?{qs}" if qs else path

        headers = {**self._org_headers(), **self._auth_headers(method, signed_path, body)}
        client = await self.client()
        resp = await client.request(
            method, self.cfg.api_url(path), params=params, content=body or None, headers=headers
        )
        if not resp.is_success:
            raise JumpServerError(
                f"{method} {path} -> {resp.status_code}: {resp.text[:500]}"
            )
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {"_raw": resp.text}

    # --- auth ------------------------------------------------------------------

    async def login_with_password(self, username: str, password: str) -> str:
        """Session login; stores a bearer token for subsequent calls."""
        client = await self.client()
        resp = await client.post(
            self.cfg.api_url("authentication/auth/"),
            json={"username": username, "password": password},
            headers=self._org_headers(),
        )
        if not resp.is_success:
            raise JumpServerError(f"login failed: {resp.status_code} {resp.text[:300]}")
        data = resp.json()
        token = data.get("token")
        if not token:
            raise JumpServerError(f"login returned no token: {data!r}")
        self._token = token
        return token

    async def ping(self) -> dict:
        """Unauthenticated health check (works before credentials are configured)."""
        client = await self.client()
        resp = await client.get(f"{self.cfg.url.rstrip('/')}/api/health/")
        resp.raise_for_status()
        return resp.json()

    # --- discovery -------------------------------------------------------------

    async def list_assets(self, *, search: str | None = None, limit: int = 200) -> list[dict]:
        """Assets visible to the current account (permission-filtered by JumpServer)."""
        params: dict = {"limit": limit}
        if search:
            params["search"] = search
        data = await self.request("GET", "assets/assets/", params=params)
        return data.get("results", data if isinstance(data, list) else [])

    async def get_asset(self, asset_id: str) -> dict:
        return await self.request("GET", f"assets/assets/{asset_id}/")

    async def list_accounts(self, asset_id: str) -> list[dict]:
        """Accounts (system users) bound to an asset — the ``#account`` in a login."""
        data = await self.request(
            "GET", "accounts/accounts/", params={"asset": asset_id, "limit": 200}
        )
        return data.get("results", [])

    # --- audit -----------------------------------------------------------------

    async def command_records(
        self, *, asset: str | None = None, user: str | None = None, limit: int = 50
    ) -> list[dict]:
        """Recent command records — the answer to 'what did the agent just run?'."""
        params: dict = {"limit": limit, "order": "-timestamp"}
        if asset:
            params["asset"] = asset
        if user:
            params["user"] = user
        data = await self.request("GET", "audits/commands/", params=params)
        return data.get("results", [])

    async def login_records(self, *, limit: int = 50) -> list[dict]:
        data = await self.request(
            "GET", "audits/login-logs/", params={"limit": limit, "order": "-timestamp"}
        )
        return data.get("results", [])

    # --- tickets ---------------------------------------------------------------

    async def create_ticket(
        self, *, title: str, ticket_type: str, body: str = "", assignees: list[str] | None = None
    ) -> dict:
        """Raise an approval ticket (e.g. a command or login-approval request)."""
        payload = {
            "title": title,
            "type": ticket_type,
            "body": body,
            "assignees": assignees or [],
        }
        return await self.request("POST", "tickets/tickets/", json=payload)

    async def list_tickets(self, *, state: str | None = None, limit: int = 50) -> list[dict]:
        params: dict = {"limit": limit, "order": "-date_created"}
        if state:
            params["state"] = state
        data = await self.request("GET", "tickets/tickets/", params=params)
        return data.get("results", [])
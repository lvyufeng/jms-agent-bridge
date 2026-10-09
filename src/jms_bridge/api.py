"""JumpServer REST control plane.

Used for what the API is genuinely good at: discovering the assets and accounts an
account may reach, reading the audit trail, and raising approval tickets. Anything
that *executes* on a host goes through the SSH data plane instead (see
:mod:`jms_bridge.ssh`).

Authentication
--------------
JumpServer accepts several schemes (DRF authenticator order, see
``apps/jumpserver/settings/libs.py``); this client supports the two that are stable
and unattended:

* **Private token** — ``Authorization: Token <token>``
* **Access key** — the *HTTP Signature* scheme (``pyhttpsig``), NOT a plain HMAC
  header. The signed string is::

      (request-target): <method> <path>\\n
      date: <RFC1123 date>\\n
      accept: application/json

  (plus ``x-jms-org`` when an org is set), signed with HMAC-SHA256 and the secret,
  base64-encoded, and placed in::

      Authorization: Signature keyId="<id>",algorithm="hmac-sha256",
                     headers="(request-target) date accept",signature="<b64>"

Session login (username/password → bearer token) is available via
:meth:`JumpServerAPI.login_with_password` but the bridge prefers a long-lived token
or access key so nothing asks for a password at runtime.

Endpoint facts verified against ``jumpserver/jumpserver@dev``:

* pagination is ``limit`` / ``offset``; ordering is ``order``; global search is ``q``
* asset list for the current user: ``GET /api/v1/perms/users/my/assets/``
* command audit: ``GET /api/v1/terminal/commands/`` (default window: last 5 days)
* tickets are created via the ``open`` action, not a plain ``POST`` on the collection
* product version (unauthenticated): ``GET /api/v1/settings/public/open/``
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import format_datetime

import httpx

from .config import BastionConfig

log = logging.getLogger("jms_bridge.api")

#: Headers bound into the signature, in the order JumpServer's verifier expects.
SIGNED_HEADERS = ["(request-target)", "date", "accept"]


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
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- auth ------------------------------------------------------------------

    def sign_access_key(
        self, method: str, path: str, *, date: str, accept: str, org: str | None
    ) -> tuple[str, dict[str, str]]:
        """Return ``(authorization_header, headers_to_send)`` for an access-key request.

        Mirrors ``pyhttpsig``'s ``HeaderSigner`` exactly (see
        ``httpsig.utils.generate_message`` and ``build_signature_template``): the
        signed lines are ``<header>: <value>`` joined by ``\\n``, with
        ``(request-target): <method> <path>`` first.
        """
        assert self.cfg.access_key_id and self.cfg.access_key_secret

        headers = {"date": date, "accept": accept}
        signed = list(SIGNED_HEADERS)
        if org:
            headers["x-jms-org"] = org
            signed = [*SIGNED_HEADERS, "x-jms-org"]

        lines = []
        for h in signed:
            if h == "(request-target)":
                lines.append(f"(request-target): {method.lower()} {path}")
            else:
                lines.append(f"{h}: {headers[h]}")
        signable = "\n".join(lines).encode("ascii")

        signature = base64.b64encode(
            hmac.new(
                self.cfg.access_key_secret.encode("ascii"), signable, hashlib.sha256
            ).digest()
        ).decode("ascii")

        authorization = (
            f'Signature keyId="{self.cfg.access_key_id}",'
            f'algorithm="hmac-sha256",signature="{signature}",'
            f'headers="{" ".join(signed)}"'
        )
        return authorization, headers

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Build the auth headers for one request, choosing the configured scheme."""
        if self.cfg.access_key_id and self.cfg.access_key_secret:
            date = format_datetime(datetime.now(timezone.utc))
            auth, extra = self.sign_access_key(
                method, path, date=date, accept="application/json", org=self.cfg.org_id
            )
            return {"Authorization": auth, "Date": extra["date"], **{
                k.title(): v for k, v in extra.items() if k in ("accept", "x-jms-org")
            }}
        # Precedence: a live session token, then a cached session token, then a
        # long-lived private token. Session tokens use "Bearer"; private tokens use
        # "Token". Access-key signing is handled above.
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        if self.cfg.session_token:
            return {"Authorization": f"Bearer {self.cfg.session_token}"}
        if self.cfg.private_token:
            return {"Authorization": f"Token {self.cfg.private_token}"}
        raise JumpServerError(
            "No JumpServer credentials configured. Run `jms-bridge login`, or set "
            "JMS_BRIDGE_SESSION_TOKEN / JMS_BRIDGE_PRIVATE_TOKEN / "
            "JMS_BRIDGE_ACCESS_KEY_ID + JMS_BRIDGE_ACCESS_KEY_SECRET."
        )

    async def request(
        self, method: str, path: str, *, params: dict | None = None, json: dict | None = None
    ) -> dict:
        url = self.cfg.api_url(path)
        # The signature binds the *request target* exactly as it goes on the wire:
        # the full path including the API base prefix and the query string
        # (JumpServer verifies against ``request.get_full_path()``). Signing the
        # bare relative path silently fails every request with "Invalid signature".
        signed_path = url.split(self.cfg.url.rstrip("/"), 1)[-1] or "/"
        if params:
            qs = str(httpx.QueryParams(params))
            signed_path = f"{signed_path}?{qs}" if qs else signed_path

        headers = self._auth_headers(method, signed_path)
        body = None
        if json is not None:
            body = __import__("json").dumps(json).encode()
            headers["Content-Type"] = "application/json"
        client = await self.client()
        resp = await client.request(
            method, url, params=params, content=body, headers=headers
        )
        if not resp.is_success:
            raise JumpServerError(f"{method} {path} -> {resp.status_code}: {resp.text[:500]}")
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {"_raw": resp.text}

    async def login_with_password(self, username: str, password: str) -> str:
        """Session login; stores a bearer token for subsequent calls."""
        client = await self.client()
        resp = await client.post(
            self.cfg.api_url("authentication/auth/"),
            json={"username": username, "password": password},
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

    async def version(self) -> str | None:
        """Product version, from the unauthenticated public-settings endpoint."""
        try:
            data = await self.request("GET", "settings/public/open/")
        except JumpServerError:
            return None
        interface = data.get("INTERFACE") or {}
        return interface.get("version") or None

    # --- discovery -------------------------------------------------------------

    async def list_assets(self, *, search: str | None = None, limit: int = 200) -> list[dict]:
        """Assets the current account is permitted to reach (the perm-scoped list)."""
        params: dict = {"limit": limit}
        if search:
            params["q"] = search
        data = await self.request("GET", "perms/users/my/assets/", params=params)
        return data.get("results", data if isinstance(data, list) else [])

    async def get_asset(self, asset_id: str) -> dict:
        """One permitted asset, including its ``permed_accounts`` and protocols.

        This single call carries everything needed to build a direct-login string.
        """
        return await self.request("GET", f"perms/users/my/assets/{asset_id}/")

    async def list_accounts(self, asset_id: str) -> list[dict]:
        """Accounts (system users) bound to an asset — the ``#account`` in a login."""
        data = await self.request(
            "GET", "accounts/accounts/", params={"asset": asset_id, "limit": 200}
        )
        return data.get("results", [])

    # --- audit -----------------------------------------------------------------

    async def command_records(
        self,
        *,
        asset_id: str | None = None,
        user: str | None = None,
        days: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Recent command records — the answer to 'what did the agent just run?'"""
        params: dict = {"limit": limit, "order": "-timestamp"}
        if asset_id:
            params["asset_id"] = asset_id
        if user:
            params["user"] = user
        if days:
            params["days"] = days
        data = await self.request("GET", "terminal/commands/", params=params)
        return data.get("results", [])

    async def login_records(self, *, limit: int = 50) -> list[dict]:
        data = await self.request(
            "GET", "audits/login-logs/", params={"limit": limit, "order": "-datetime"}
        )
        return data.get("results", [])

    # --- tickets ---------------------------------------------------------------

    #: Ticket subtypes map to their own ``apply-*`` collections; creation is the
    #: ``open`` action on that collection, not a POST to the collection itself.
    TICKET_TYPES = {
        "login_asset_confirm": "apply-login-asset-tickets",
        "command_confirm": "apply-command-tickets",
        "apply_asset": "apply-asset-tickets",
        "login_confirm": "apply-login-tickets",
    }

    async def create_ticket(
        self,
        *,
        title: str,
        ticket_type: str = "login_asset_confirm",
        org_id: str | None = None,
        comment: str = "",
        **fields: object,
    ) -> dict:
        """Raise an approval ticket via the subtype's ``open`` action.

        ``fields`` carries the type-specific payload, e.g. for a login-asset ticket:
        ``apply_login_user``, ``apply_login_asset``, ``apply_login_account``.
        """
        collection = self.TICKET_TYPES.get(ticket_type)
        if collection is None:
            raise JumpServerError(
                f"unknown ticket type {ticket_type!r}; known: {sorted(self.TICKET_TYPES)}"
            )
        payload = {
            "title": title,
            "type": ticket_type,
            "org_id": org_id or self.cfg.org_id,
            "comment": comment,
            **fields,
        }
        return await self.request("POST", f"tickets/{collection}/open/", json=payload)

    async def list_tickets(self, *, state: str | None = None, limit: int = 50) -> list[dict]:
        params: dict = {"limit": limit, "order": "-date_created"}
        if state:
            params["state"] = state
        data = await self.request("GET", "tickets/tickets/", params=params)
        return data.get("results", [])


__all__ = ["JumpServerAPI", "JumpServerError", "SIGNED_HEADERS", "json"]
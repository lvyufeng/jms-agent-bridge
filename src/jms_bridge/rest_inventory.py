"""An atomic, hourly snapshot of permitted REST assets, independent of hosts.yaml."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import time
import uuid

import httpx

from .api import JumpServerAPI, JumpServerError
from .inventory import Host, Inventory, InventoryError
from .results import atomic_json

log = logging.getLogger(__name__)


class RESTInventory(Inventory):
    def resolve(self, selectors):
        try:
            return super().resolve(selectors)
        except InventoryError:
            raise InventoryError(
                'target is not in the REST snapshot; use discover(topic="assets", query=...)'
            ) from None


class AssetCatalog:
    def __init__(self, cfg, settings):
        self.cfg, self.settings = cfg, settings
        self.state = settings.state_dir
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.interval = settings.refresh_interval
        # Do not reuse another endpoint, organization, or credential's permissions.
        identity = [
            cfg.url,
            cfg.org_id,
            cfg.login_user,
            cfg.access_key_id,
            cfg.access_key_secret,
            cfg.private_token,
            cfg.session_token,
            settings.preferred_account,
        ]
        self.identity = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        self.snapshot = {"refreshed_at": 0, "assets": [], "identity": self.identity}
        self.inventory = RESTInventory({}, {})
        self.error = None
        self._lock = asyncio.Lock()
        self._last_attempt = 0.0
        self._task = None
        self.cache = self.state / "assets.json"
        if self.cache.exists():
            try:
                snapshot = json.loads(self.cache.read_text())
                if snapshot["identity"] == self.identity:
                    inventory = self._build(snapshot)
                    self.snapshot, self.inventory = snapshot, inventory
            except (ValueError, KeyError, TypeError, OSError):
                self.error = "cached asset snapshot is invalid"

    def make_api(self):
        api = JumpServerAPI(self.cfg)
        if self.settings.rest_proxy:
            api._client = httpx.AsyncClient(
                proxy=self.settings.rest_proxy,
                timeout=self.cfg.api_timeout,
                verify=self.cfg.verify_tls,
                trust_env=False,
            )
        return api

    def explain_no_targets(self) -> str | None:
        """Why the snapshot has assets but no SSH target, when that is the cause.

        ``run`` reports "no permitted SSH assets matched the targets" for any empty
        match, but the most common reason in a fresh deployment is not the selector:
        every asset offers several permitted accounts, ``preferred_account`` is unset,
        and the account is deliberately left unresolved — an ambiguity the snapshot
        refuses to guess. The assets are all there and every selector matches nothing,
        so the generic message sends the reader looking in the wrong place.
        """
        if self.inventory.hosts or not self.snapshot["assets"]:
            return None
        ssh = [a for a in self.snapshot["assets"] if a["active"] and "ssh" in a["protocols"]]
        if not ssh:
            return (
                f"{len(self.snapshot['assets'])} assets are visible, but none is active with "
                "an SSH protocol permitted; check the asset/protocol permissions."
            )
        accounts = {a["account"] for a in ssh}
        if accounts == {None}:
            hint = (
                "set bridge.preferred_account in config.toml to one the assets permit "
                "(for example \"root\")"
                if not self.settings.preferred_account
                else (
                    f"bridge.preferred_account is {self.settings.preferred_account!r}, which "
                    "none of these assets permits"
                )
            )
            offered = sorted(
                {len([x for x in a["accounts"] if x]) for a in ssh if a.get("accounts")}
            )
            plural = f" (typically {offered[0]})" if offered else ""
            return (
                f"{len(ssh)} active SSH assets are visible, but every one offers several "
                f"permitted accounts{plural} and no account is selected, so none becomes a "
                f"target: {hint}."
            )
        return None

    def status(self):
        refreshed = self.snapshot["refreshed_at"]
        return {
            "source": "JumpServer REST",
            "refresh_interval_seconds": self.interval,
            "last_success": refreshed or None,
            "next_refresh": refreshed + self.interval if refreshed else None,
            "stale": not refreshed or time.time() - refreshed >= self.interval,
            "assets": len(self.snapshot["assets"]),
            "ssh_targets": len(self.inventory.hosts),
            "last_error": self.error,
            "refresh_in_progress": self._lock.locked(),
        }

    def _build(self, snapshot):
        hosts, selectors, name_counts, address_counts = {}, {}, {}, {}
        for asset in snapshot["assets"]:
            name_counts[asset["name"]] = name_counts.get(asset["name"], 0) + 1
            address_counts[asset["address"]] = address_counts.get(asset["address"], 0) + 1
        for asset in snapshot["assets"]:
            if not asset["active"] or not asset["account"] or "ssh" not in asset["protocols"]:
                continue
            asset_id = str(uuid.UUID(asset["id"]))
            login, account = asset["login"], asset["account"]
            if not login or not re.fullmatch(r"[A-Za-z0-9_.@-]+", login):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", account):
                continue
            # UUIDs are canonical, so display names can never overwrite a target.
            hosts[asset_id] = Host(
                name=asset_id,
                login=login,
                account=account,
                asset_id=asset_id,
                address=asset["address"],
                platform="generic",
            )
            for label, counts in [(asset["name"], name_counts), (asset["address"], address_counts)]:
                if label and label not in ("all", "*") and counts[label] == 1:
                    selectors.setdefault(label, []).append(asset_id)
        groups = {
            k: list(set(v)) for k, v in selectors.items() if len(set(v)) == 1 and k not in hosts
        }
        return RESTInventory(hosts, groups)

    async def _fetch(self):
        api = self.make_api()
        try:
            assets, truncated = await api.list_assets()
            if truncated:
                raise ValueError("REST returned a partial asset list")
            semaphore = asyncio.Semaphore(4)

            async def detail(asset):
                async with semaphore:
                    try:
                        data = await api.get_asset(asset["id"])
                    except JumpServerError as exc:
                        if " -> 403:" in str(exc) or " -> 404:" in str(exc):
                            return None  # Permission changed during refresh.
                        raise
                usable = list(
                    dict.fromkeys(
                        x["username"]
                        for x in data.get("permed_accounts", [])
                        if x.get("has_secret") and x.get("username")
                    )
                )
                preferred = self.settings.preferred_account
                account = (
                    (preferred if preferred in usable else None)
                    if preferred
                    else (usable[0] if len(usable) == 1 else None)
                )
                return {
                    "id": str(uuid.UUID(data["id"])),
                    "name": data.get("name") or str(data["id"]),
                    "address": data.get("address") or "",
                    "active": data.get("is_active", True),
                    "protocols": [x["name"] for x in data.get("permed_protocols", [])],
                    "accounts": usable,
                    "account": account,
                    "login": self.cfg.login_user,
                }

            tasks = [asyncio.create_task(detail(asset)) for asset in assets]
            try:
                rows = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            return [row for row in rows if row is not None]
        finally:
            await api.aclose()

    async def refresh(self):
        async with self._lock:
            self._last_attempt = time.time()
            try:
                assets = await self._fetch()
                snapshot = {
                    "refreshed_at": time.time(),
                    "assets": assets,
                    "identity": self.identity,
                }
                inventory = self._build(snapshot)
                atomic_json(self.cache, snapshot)
                self.snapshot, self.inventory, self.error = snapshot, inventory, None
                log.info(
                    "REST refresh: %d assets, %d SSH targets", len(assets), len(inventory.hosts)
                )
            except Exception as exc:
                # Never publish raw HTTP bodies, proxy URLs, or authentication material.
                self.error = (
                    f"REST refresh failed ({type(exc).__name__}); check API access/configuration"
                )
                log.warning("%s; previous snapshot retained", self.error)
            return {"ok": self.error is None, **self.status(), "refresh_in_progress": False}

    async def ensure(self):
        if not self.snapshot["refreshed_at"] and self._lock.locked():
            async with self._lock:
                pass
        if time.time() - self.snapshot["refreshed_at"] >= self.interval:
            if time.time() - self._last_attempt > 5:
                await self.refresh()
        if not self.snapshot["refreshed_at"]:
            raise ValueError(
                self.error or "REST inventory is not ready; configure API access first"
            )
        return self.inventory

    async def start(self):
        self._task = asyncio.create_task(self._background())

    async def _background(self):
        # REST unavailability must not prevent MCP initialization/status calls.
        await self.refresh()
        while True:
            await asyncio.sleep(min(30, self.interval))
            due = time.time() - self.snapshot["refreshed_at"] >= self.interval
            retry_after = min(300, self.interval)
            if (due or self.error) and time.time() - self._last_attempt >= retry_after:
                await self.refresh()

    async def close(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def search(self, query="", offset=0, limit=10):
        rows = [
            a
            for a in self.snapshot["assets"]
            if query.lower() in " ".join([a["name"], a["id"], a["address"]]).lower()
        ]
        page = [
            {
                k: a[k]
                for k in ("id", "name", "address", "active", "protocols", "accounts", "account")
            }
            for a in rows[offset : offset + limit]
        ]
        return {
            "assets": page,
            "matched": len(rows),
            "next_offset": offset + len(page) if offset + len(page) < len(rows) else None,
            "inventory": self.status(),
        }

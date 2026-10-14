# Contributing

## Layout

```
src/jms_bridge/
  config.py      config loading (TOML + env overrides)
  inventory.py   hosts.yaml -> Host/Inventory, selectors, remote-command wrapping
  platforms.py   accelerator profiles (ascend-910c / cuda / aarch64 / generic)
  ssh.py         SSH data plane: ControlMaster multiplexing, parallel fan-out
  jobs.py        remote job lifecycle (tmux + pidfile + logfile)
  api.py         JumpServer REST control plane (auth, discovery, audit, tickets)
  server.py      the MCP tool surface
  cli.py         init / doctor / probe / serve
tests/           unit tests (no network)
```

## Ground rules

- **Control plane vs data plane.** Discovery/audit/tickets go through the REST API;
  anything that runs on a host goes through SSH. Keep the split.
- **Never block a tool call.** Long work becomes a job (`jms_submit_job`), never a
  synchronous `jms_exec`.
- **Argv, not shell strings.** `subprocess` always receives a list; remote scripts are
  piped to `bash -s`. Nothing user-supplied may be interpreted by the local shell.
- **No PTY/menu emulation.** Only JumpServer's documented per-asset direct login.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

Integration is verified with `jms-bridge doctor` (read-only) against a real bastion,
not from CI. CI runs lint + unit tests only.

## Verifying a fresh clone

The first-run path is scripted, so a fresh clone can be *proved* to work rather than
assumed:

```bash
JMS_BRIDGE_PASSWORD='...' scripts/verify-onboarding.sh
```

It runs `init` → `login` → `jms_list_assets` → build `hosts.yaml` → `doctor` (the gate)
→ `jms_dev_discover` → one real `jms_dev_run`, and stops at the first step that does not
work, printing that step's actual output. The config, the inventory and the credentials
file are all pinned inside a scratch directory and the OS keyring is bypassed, so it never
touches `~/.config/jms-bridge`; `$HOME` is left alone because ssh still needs the real
`~/.ssh`. Exit status is 0 only when the cross-chip table comes back green.

`scripts/verify_support.py` and `scripts/mcp_call.py` are the helpers it drives: `mcp_call.py`
speaks the MCP wire protocol (there is no CLI subcommand for the `jms_*` tools), and
`verify_support.py` turns the `jms_list_assets` result into a `hosts.yaml` (`mkhosts`) and appends
a discovered `workspaces:` block to it (`discover`).

## Releasing

The version lives in **two** places and the changelog in a third; a release commit moves all
three together:

```bash
git switch -c release/0.2.2 origin/main
# edit CHANGELOG.md:  ## Unreleased → ## 0.2.2 — 2026-10-14
# bump both:          pyproject.toml [project] version, src/jms_bridge/__init__.py __version__
git commit -am "chore: release 0.2.2"
```

Merge that through the normal PR flow, then tag `main` and push the tag:

```bash
git switch main && git pull
git tag -a v0.2.2 -m "..."   # the tag body becomes the GitHub release notes
git push origin v0.2.2
```

`.github/workflows/release.yml` does the rest: it re-checks that `vX.Y.Z` agrees with
`pyproject.toml` and `__version__`, builds the sdist and wheel, runs `twine check`, publishes to
PyPI, and cuts the GitHub release with the artifacts attached. A mistyped tag therefore fails the
build job rather than uploading a wrong version.

Two things the workflow needs that are **repository settings, not files**:

- A `pypi` GitHub environment, which is the publish gate — add required reviewers to it if the
  upload should wait for a human.
- A PyPI [trusted publisher](https://docs.pypi.org/trusted-publishers/) registered for this repo,
  workflow `release.yml`, environment `pypi`. That is the credential: the publish job carries no
  token. To use an API token instead, set a `PYPI_API_TOKEN` secret and add
  `password: ${{ secrets.PYPI_API_TOKEN }}` to the publish step — but note that an explicit
  password *disables* the trusted-publishing path rather than falling back to it, so pick one.

`workflow_dispatch` runs the build and the checks without publishing, which is the way to find a
broken packaging manifest before spending a version number on it.

`v0.2.0` and `v0.2.1` predate this workflow, and a tag only triggers it on the push that creates
it — so those two exist as tags but were never published and have no GitHub release. Once the
trusted publisher above is registered, re-point each one at the commit it already names and push
it again to run the workflow (`git push --force origin v0.2.1`) — or publish that version by hand.
Until then the tags are only markers, as before.

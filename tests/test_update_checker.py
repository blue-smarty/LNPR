"""Tests for update checker logic."""

from __future__ import annotations

import json
from urllib.error import URLError


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


def test_is_newer_version_semver_compare():
    from src.update_checker import _is_newer_version

    assert _is_newer_version("v1.2.0", "v1.1.9")
    assert not _is_newer_version("v1.2.0", "v1.2.0")
    assert not _is_newer_version("v1.2.0", "v1.2.1")


def test_check_for_updates_available(monkeypatch):
    from src import update_checker

    def _fake_urlopen(_request, timeout):  # noqa: ANN001
        assert timeout == 1.0
        return _FakeHTTPResponse(
            {"tag_name": "v1.3.0", "html_url": "https://github.com/blue-smarty/LNPR/releases/tag/v1.3.0"}
        )

    monkeypatch.setattr(update_checker.urllib.request, "urlopen", _fake_urlopen)
    info = update_checker.check_for_updates(current_version="v1.2.0", timeout=1.0)
    assert info.error is None
    assert info.update_available
    assert info.latest_version == "v1.3.0"


def test_check_for_updates_no_update(monkeypatch):
    from src import update_checker

    def _fake_urlopen(_request, timeout):  # noqa: ANN001
        assert timeout == 1.0
        return _FakeHTTPResponse(
            {"tag_name": "v1.2.0", "html_url": "https://github.com/blue-smarty/LNPR/releases/tag/v1.2.0"}
        )

    monkeypatch.setattr(update_checker.urllib.request, "urlopen", _fake_urlopen)
    info = update_checker.check_for_updates(current_version="v1.2.0", timeout=1.0)
    assert info.error is None
    assert not info.update_available


def test_check_for_updates_network_error(monkeypatch):
    from src import update_checker

    def _fake_urlopen(_request, timeout):  # noqa: ANN001
        assert timeout == 1.0
        raise URLError("no network")

    monkeypatch.setattr(update_checker.urllib.request, "urlopen", _fake_urlopen)
    info = update_checker.check_for_updates(current_version="v1.2.0", timeout=1.0)
    assert info.error is not None
    assert not info.update_available

"""Update checker for LNPR releases."""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class UpdateInfo:
    current_version: str
    latest_version: Optional[str]
    update_available: bool
    release_url: Optional[str]
    error: Optional[str] = None


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", (version or "").strip().lstrip("vV"))
    return tuple(int(p) for p in parts) if parts else (0,)


def _is_newer_version(latest_version: str, current_version: str) -> bool:
    latest = _version_tuple(latest_version)
    current = _version_tuple(current_version)
    max_len = max(len(latest), len(current))
    latest = latest + (0,) * (max_len - len(latest))
    current = current + (0,) * (max_len - len(current))
    return latest > current


def _resolve_current_version() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "--no-pager", "describe", "--tags", "--abbrev=0"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        version = result.stdout.strip()
        if version:
            return version
    except Exception:
        pass
    return "0.0.0"


def check_for_updates(
    owner: str = "blue-smarty",
    repo: str = "LNPR",
    current_version: Optional[str] = None,
    timeout: float = 5.0,
) -> UpdateInfo:
    """Check GitHub releases and report whether a newer version exists."""
    current = current_version or _resolve_current_version()
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "LNPR-UpdateChecker",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return UpdateInfo(
            current_version=current,
            latest_version=None,
            update_available=False,
            release_url=None,
            error=f"HTTP {exc.code} while checking updates",
        )
    except urllib.error.URLError as exc:
        return UpdateInfo(
            current_version=current,
            latest_version=None,
            update_available=False,
            release_url=None,
            error=f"Network error while checking updates: {exc.reason}",
        )
    except Exception as exc:
        return UpdateInfo(
            current_version=current,
            latest_version=None,
            update_available=False,
            release_url=None,
            error=f"Failed to check updates: {exc}",
        )

    latest = payload.get("tag_name") or payload.get("name")
    release_url = payload.get("html_url")
    if not latest:
        return UpdateInfo(
            current_version=current,
            latest_version=None,
            update_available=False,
            release_url=release_url,
            error="Latest release did not include a version tag",
        )

    return UpdateInfo(
        current_version=current,
        latest_version=latest,
        update_available=_is_newer_version(latest, current),
        release_url=release_url,
        error=None,
    )


"""
Minimal Suno API client for prompt-to-track generation.

This module is intentionally playback-agnostic: it returns URLs and metadata
that an external orchestrator can play later.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.sunoapi.org"
DEFAULT_MODEL = "V4_5ALL"
SUCCESS_STATUSES = {"SUCCESS"}
FAILURE_STATUSES = {
    "CREATE_TASK_FAILED",
    "GENERATE_AUDIO_FAILED",
    "CALLBACK_EXCEPTION",
    "SENSITIVE_WORD_ERROR",
}


class SunoError(RuntimeError):
    """Raised when Suno API requests fail or return an error response."""


class SunoTimeoutError(SunoError):
    """Raised when waiting for a task exceeds the configured timeout."""


@dataclass(frozen=True)
class SunoTrack:
    """Normalized per-track data from Suno generation details."""

    id: str
    title: str
    stream_url: Optional[str]
    audio_url: Optional[str]
    image_url: Optional[str]
    prompt: Optional[str]
    model_name: Optional[str]
    duration_s: Optional[float]
    tags: Optional[str]
    created_at: Optional[str]


@dataclass(frozen=True)
class SunoGenerationResult:
    """Task-level generation result suitable for downstream orchestration."""

    task_id: str
    status: str
    tracks: List[SunoTrack]
    raw: Dict[str, Any]


def _read_env_file(path: Path = Path(".env")) -> Dict[str, str]:
    """Best-effort local .env parser for simple KEY=VALUE lines."""
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        values[key] = value
    return values


def get_suno_api_key(explicit_api_key: Optional[str] = None) -> str:
    """Resolve Suno API key from arg -> env -> .env."""
    if explicit_api_key:
        return explicit_api_key

    env_key = os.environ.get("SUNO_API_KEY")
    if env_key:
        return env_key

    env_file_key = _read_env_file().get("SUNO_API_KEY")
    if env_file_key:
        return env_file_key

    raise SunoError("SUNO_API_KEY not found (arg/env/.env).")


def _request_json(
    method: str,
    path: str,
    *,
    api_key: str,
    base_url: str,
    query: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"

    payload: Optional[bytes] = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")

    req = Request(
        url=url,
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        details = err.read().decode("utf-8", errors="replace")
        raise SunoError(f"Suno HTTP error {err.code}: {details}") from err
    except URLError as err:
        raise SunoError(f"Suno connection error: {err}") from err


def create_generation_task(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    instrumental: bool = False,
    custom_mode: bool = False,
    callback_url: Optional[str] = None,
    timeout_s: float = 30.0,
) -> str:
    """
    Create a Suno generation task and return its task ID.

    Uses non-custom mode by default (prompt-only), which is the simplest flow.
    """
    key = get_suno_api_key(api_key)
    body: Dict[str, Any] = {
        "customMode": custom_mode,
        "instrumental": instrumental,
        "model": model,
        "prompt": prompt,
    }
    if callback_url:
        body["callBackUrl"] = callback_url

    data = _request_json(
        "POST",
        "/api/v1/generate",
        api_key=key,
        base_url=base_url,
        body=body,
        timeout_s=timeout_s,
    )
    code = data.get("code")
    if code != 200:
        raise SunoError(f"Suno create task failed: code={code}, msg={data.get('msg')}")
    task_id = ((data.get("data") or {}).get("taskId") or "").strip()
    if not task_id:
        raise SunoError("Suno create task response missing taskId.")
    return task_id


def get_generation_details(
    task_id: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Fetch raw generation details for a task ID."""
    key = get_suno_api_key(api_key)
    data = _request_json(
        "GET",
        "/api/v1/generate/record-info",
        api_key=key,
        base_url=base_url,
        query={"taskId": task_id},
        timeout_s=timeout_s,
    )
    code = data.get("code")
    if code != 200:
        raise SunoError(f"Suno details failed: code={code}, msg={data.get('msg')}")
    return data


def _parse_tracks(details_data: Dict[str, Any]) -> List[SunoTrack]:
    root_data = details_data.get("data") or {}
    response = root_data.get("response") or {}
    items = response.get("sunoData") or []
    tracks: List[SunoTrack] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tracks.append(
            SunoTrack(
                id=str(item.get("id") or ""),
                title=str(item.get("title") or ""),
                stream_url=item.get("streamAudioUrl"),
                audio_url=item.get("audioUrl"),
                image_url=item.get("imageUrl"),
                prompt=item.get("prompt"),
                model_name=item.get("modelName"),
                duration_s=float(item["duration"]) if item.get("duration") else None,
                tags=item.get("tags"),
                created_at=item.get("createTime"),
            )
        )
    return tracks


def wait_for_generation(
    task_id: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    poll_interval_s: float = 3.0,
    timeout_s: float = 240.0,
) -> SunoGenerationResult:
    """Poll task status until success/failure/timeout, then return tracks."""
    start = time.time()
    while True:
        details = get_generation_details(
            task_id, api_key=api_key, base_url=base_url, timeout_s=30.0
        )
        data = details.get("data") or {}
        status = str(data.get("status") or "UNKNOWN")

        if status in SUCCESS_STATUSES:
            return SunoGenerationResult(
                task_id=task_id,
                status=status,
                tracks=_parse_tracks(details),
                raw=details,
            )

        if status in FAILURE_STATUSES:
            raise SunoError(
                "Suno generation failed: "
                f"status={status}, errorCode={data.get('errorCode')}, "
                f"errorMessage={data.get('errorMessage')}"
            )

        if time.time() - start >= timeout_s:
            raise SunoTimeoutError(
                f"Timed out waiting for task {task_id} after {timeout_s:.1f}s "
                f"(last_status={status})."
            )

        time.sleep(max(0.1, poll_interval_s))


def generate_music(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    instrumental: bool = False,
    custom_mode: bool = False,
    callback_url: Optional[str] = None,
    wait: bool = True,
    poll_interval_s: float = 3.0,
    timeout_s: float = 240.0,
) -> SunoGenerationResult | str:
    """
    End-to-end helper.

    - If wait=False: returns task_id (str).
    - If wait=True: returns SunoGenerationResult with track URLs/metadata.
    """
    task_id = create_generation_task(
        prompt=prompt,
        api_key=api_key,
        base_url=base_url,
        model=model,
        instrumental=instrumental,
        custom_mode=custom_mode,
        callback_url=callback_url,
    )
    if not wait:
        return task_id
    return wait_for_generation(
        task_id,
        api_key=api_key,
        base_url=base_url,
        poll_interval_s=poll_interval_s,
        timeout_s=timeout_s,
    )


def generate_from_prompt(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    wait: bool = True,
    **kwargs: Any,
) -> SunoGenerationResult | str:
    """
    Orchestrator-friendly alias.

    - `wait=False` returns task_id to resolve later.
    - `wait=True` returns final generation result with track URLs.
    """
    return generate_music(prompt, api_key=api_key, wait=wait, **kwargs)

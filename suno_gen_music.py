"""Suno TreeHacks API client for prompt-to-track generation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://studio-api.prod.suno.com"
GENERATE_PATH = "/api/v2/external/hackathons/generate"
CLIPS_PATH = "/api/v2/external/hackathons/clips"
SUCCESS_STATUSES = {"complete", "streaming"}
FAILURE_STATUSES = {"failed", "error", "rejected", "timeout", "cancelled"}


class SunoError(RuntimeError):
    """Raised when Suno API requests fail or return an error response."""


class SunoTimeoutError(SunoError):
    """Raised when waiting for a task exceeds the configured timeout."""


@dataclass(frozen=True)
class SunoTrack:
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
    task_id: str
    status: str
    tracks: List[SunoTrack]
    raw: Dict[str, Any]


def _read_env_file(path: Path = Path(".env")) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def get_suno_api_key(explicit_api_key: Optional[str] = None) -> str:
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
    body: Optional[Dict[str, Any]] = None,
    timeout_s: float = 30.0,
) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(
        url=url,
        data=payload,
        method=method,
        headers={
            # TreeHacks docs specify Bearer auth; keep api-key for compatibility.
            "Authorization": f"Bearer {api_key}",
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "treehacks26/0.1 (+python urllib)",
        },
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as err:
        details = err.read().decode("utf-8", errors="replace")
        raise SunoError(f"Suno HTTP error {err.code}: {details}") from err
    except URLError as err:
        raise SunoError(f"Suno connection error: {err}") from err


def _normalize_clips(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("clips"), list):
            return [p for p in payload["clips"] if isinstance(p, dict)]
        if isinstance(payload.get("data"), list):
            return [p for p in payload["data"] if isinstance(p, dict)]
        if payload.get("id"):
            return [payload]
    return []


def create_generation_task(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = "",
    instrumental: bool = True,
    custom_mode: bool = False,
    callback_url: Optional[str] = None,
    tags: str = "",
    negative_tags: str = "",
    lyrics_prompt: str = "",
    cover_clip_id: str = "",
    timeout_s: float = 30.0,
) -> str:
    key = get_suno_api_key(api_key)
    effective_base_url = os.environ.get("SUNO_BASE_URL", base_url).strip()
    # Force instrumental generation to disable vocals/lyrics.
    body: Dict[str, Any] = {
        "topic": prompt,
        "tags": tags,
        "make_instrumental": True,
    }
    if negative_tags:
        body["negative_tags"] = negative_tags
    if lyrics_prompt:
        body["prompt"] = lyrics_prompt
    if cover_clip_id:
        body["cover_clip_id"] = cover_clip_id
    if custom_mode:
        body["custom_mode"] = True
    if callback_url:
        body["callback_url"] = callback_url
    if model:
        body["model"] = model

    response = _request_json(
        "POST",
        GENERATE_PATH,
        api_key=key,
        base_url=effective_base_url,
        body=body,
        timeout_s=timeout_s,
    )
    clips = _normalize_clips(response)
    if not clips:
        raise SunoError(f"Suno create task returned no clips: {response}")
    task_id = str(clips[0].get("id") or "").strip()
    if not task_id:
        raise SunoError(f"Suno create task response missing clip id: {response}")
    return task_id


def get_generation_details(
    task_id: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    key = get_suno_api_key(api_key)
    effective_base_url = os.environ.get("SUNO_BASE_URL", base_url).strip()
    response = _request_json(
        "GET",
        f"{CLIPS_PATH}?ids={task_id}",
        api_key=key,
        base_url=effective_base_url,
        timeout_s=timeout_s,
    )
    return {"clips": _normalize_clips(response), "raw": response}


def _parse_tracks(details_data: Dict[str, Any]) -> List[SunoTrack]:
    items = details_data.get("clips") or []
    tracks: List[SunoTrack] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        tracks.append(
            SunoTrack(
                id=str(item.get("id") or ""),
                title=str(item.get("title") or item.get("display_name") or ""),
                stream_url=item.get("audio_url"),
                audio_url=item.get("audio_url"),
                image_url=item.get("image_url"),
                prompt=item.get("prompt") or metadata.get("prompt"),
                model_name=item.get("model_name"),
                duration_s=float(item["duration"]) if item.get("duration") else None,
                tags=item.get("tags"),
                created_at=item.get("created_at"),
            )
        )
    return tracks


def wait_for_generation(
    task_id: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    poll_interval_s: float = 2.5,
    timeout_s: float = 240.0,
) -> SunoGenerationResult:
    start = time.time()
    while True:
        details = get_generation_details(
            task_id, api_key=api_key, base_url=base_url, timeout_s=30.0
        )
        clips = details.get("clips") or []
        clip = clips[0] if clips else {}
        status = str(clip.get("status") or "unknown").lower()
        has_audio = bool(clip.get("audio_url"))

        if status in SUCCESS_STATUSES and has_audio:
            return SunoGenerationResult(
                task_id=task_id,
                status=status,
                tracks=_parse_tracks(details),
                raw=details.get("raw") or details,
            )
        if status in FAILURE_STATUSES:
            raise SunoError(f"Suno generation failed: status={status}, details={clip}")
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
    model: str = "",
    instrumental: bool = True,
    custom_mode: bool = False,
    callback_url: Optional[str] = None,
    tags: str = "",
    negative_tags: str = "",
    lyrics_prompt: str = "",
    cover_clip_id: str = "",
    wait: bool = True,
    poll_interval_s: float = 2.5,
    timeout_s: float = 240.0,
) -> SunoGenerationResult | str:
    task_id = create_generation_task(
        prompt=prompt,
        api_key=api_key,
        base_url=base_url,
        model=model,
        instrumental=instrumental,
        custom_mode=custom_mode,
        callback_url=callback_url,
        tags=tags,
        negative_tags=negative_tags,
        lyrics_prompt=lyrics_prompt,
        cover_clip_id=cover_clip_id,
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
    return generate_music(prompt, api_key=api_key, wait=wait, **kwargs)

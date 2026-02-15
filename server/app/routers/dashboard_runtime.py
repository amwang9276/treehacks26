from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


router = APIRouter(prefix="/dashboard/runtime", tags=["dashboard"])


class StartRuntimeRequest(BaseModel):
    generate: bool = True


def _runtime(request: Request):
    return request.app.state.dashboard_runtime


@router.post("/start")
def start_runtime(payload: StartRuntimeRequest, request: Request) -> Dict[str, Any]:
    runtime = _runtime(request)
    return runtime.start(generate=payload.generate)


@router.post("/stop")
def stop_runtime(request: Request) -> Dict[str, Any]:
    runtime = _runtime(request)
    return runtime.stop()


@router.get("/logs")
def runtime_logs(request: Request) -> Dict[str, Any]:
    runtime = _runtime(request)
    return runtime.logs_snapshot()


@router.get("/stream")
def runtime_stream(
    request: Request,
    fps: float = Query(default=15.0, ge=1.0, le=30.0),
) -> StreamingResponse:
    runtime = _runtime(request)
    return StreamingResponse(
        runtime.mjpeg_stream(fps=fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


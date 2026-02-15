from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

from mulan import DEFAULT_MULAN_MODEL_ID, MuLanEmbedError, MuLanEmbedder


def _resolve_audio_path(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return path

    candidates = sorted(glob.glob("songs/*.mp3"))
    if not candidates:
        raise FileNotFoundError("No mp3 file found in songs/. Pass --audio-path explicitly.")
    return Path(candidates[0]).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MuLan embedding on a local audio file.")
    parser.add_argument("--audio-path", default=None, help="Path to an audio file (mp3/wav/etc).")
    parser.add_argument("--model-id", default=os.environ.get("MULAN_MODEL_ID", DEFAULT_MULAN_MODEL_ID))
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--preview-values", type=int, default=12, help="How many vector values to print")
    args = parser.parse_args()

    try:
        audio_path = _resolve_audio_path(args.audio_path)
        embedder = MuLanEmbedder(model_id=args.model_id, device=args.device)
        vector = embedder.embed_audio_file(str(audio_path))
    except (FileNotFoundError, MuLanEmbedError) as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"[ERROR] Unexpected failure: {err}", file=sys.stderr)
        return 1

    values_n = max(1, min(int(args.preview_values), int(vector.shape[0])))
    preview = np.array2string(vector[:values_n], precision=6, separator=", ")
    print(f"audio_file: {audio_path}")
    print(f"model_id: {args.model_id}")
    print(f"device: {args.device}")
    print(f"embedding_dim: {int(vector.shape[0])}")
    print(f"l2_norm: {float(np.linalg.norm(vector)):.6f}")
    print(f"embedding_preview_first_{values_n}: {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

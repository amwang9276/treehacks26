from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import librosa
import numpy as np
import torch


DEFAULT_MULAN_MODEL_ID = "OpenMuQ/MuQ-MuLan-large"


class MuLanEmbedError(RuntimeError):
    """Raised when MuLan model load or embedding fails."""


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return vec
    return vec / norm


@dataclass
class MuLanEmbedder:
    model_id: str = DEFAULT_MULAN_MODEL_ID
    device: str = "cpu"
    sample_rate: int = 24000

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._backend = ""
        self._mulan = None
        self._processor = None
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        local_only = os.environ.get("HF_LOCAL_FILES_ONLY", "").lower() in {"1", "true", "yes"}
        load_errors: List[str] = []
        is_openmuq_model = self.model_id.startswith("OpenMuQ/")

        # Preferred backend from OpenMuQ examples.
        try:
            from muq import MuQMuLan  # type: ignore

            self._mulan = MuQMuLan.from_pretrained(
                self.model_id,
                token=hf_token,
                local_files_only=local_only,
            ).to(self.device).eval()
            self._backend = "muq"
            return
        except ModuleNotFoundError as err:
            load_errors.append(f"muq backend failed: {err}")
            if is_openmuq_model:
                raise MuLanEmbedError(
                    f"Model '{self.model_id}' requires the `muq` package. "
                    "Install it in this environment: `pip install muq`."
                ) from err
        except Exception as err:
            load_errors.append(f"muq backend failed: {err}")

        # Fallback to generic HF loading when muq package is unavailable.
        try:
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=hf_token,
                local_files_only=local_only,
            )
            self._model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=hf_token,
                local_files_only=local_only,
            ).to(self.device).eval()
            self._backend = "hf"
            return
        except Exception as err:
            load_errors.append(f"transformers backend failed: {err}")
            raise MuLanEmbedError(
                f"Failed to load MuLan model '{self.model_id}'. "
                "Ensure model exists and is accessible, install dependencies, and verify Hugging Face auth/network. "
                "If private/gated, run `huggingface-cli login` or set HF_TOKEN/HUGGINGFACE_TOKEN. "
                "If running offline, set MULAN_MODEL_ID to a local path and HF_LOCAL_FILES_ONLY=true. "
                f"Details: {' | '.join(load_errors)}"
            ) from err

    def _embed_audio_muq(self, wav: np.ndarray) -> np.ndarray:
        assert self._mulan is not None
        with torch.no_grad():
            wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(self.device)
            out = self._mulan(wavs=wav_t)
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().numpy()[0]
        if isinstance(out, dict):
            for key in ("audio_embeds", "audio_embeddings", "embeddings"):
                if key in out:
                    return out[key].detach().cpu().numpy()[0]
        raise MuLanEmbedError("Unexpected MuQ MuLan audio output format.")

    def _embed_text_muq(self, text: str) -> np.ndarray:
        assert self._mulan is not None
        with torch.no_grad():
            out = self._mulan(texts=[text])
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().numpy()[0]
        if isinstance(out, dict):
            for key in ("text_embeds", "text_embeddings", "embeddings"):
                if key in out:
                    return out[key].detach().cpu().numpy()[0]
        raise MuLanEmbedError("Unexpected MuQ MuLan text output format.")

    def _embed_audio_hf(self, wav: np.ndarray) -> np.ndarray:
        assert self._processor is not None and self._model is not None
        inputs = self._processor(
            audios=wav, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        for key in ("audio_embeds", "audio_embeddings", "pooler_output"):
            tensor = getattr(out, key, None) if not isinstance(out, dict) else out.get(key)
            if tensor is not None:
                return tensor.detach().cpu().numpy()[0]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state.mean(dim=1).detach().cpu().numpy()[0]
        raise MuLanEmbedError("Unexpected HF audio output format.")

    def _embed_text_hf(self, text: str) -> np.ndarray:
        assert self._processor is not None and self._model is not None
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        for key in ("text_embeds", "text_embeddings", "pooler_output"):
            tensor = getattr(out, key, None) if not isinstance(out, dict) else out.get(key)
            if tensor is not None:
                return tensor.detach().cpu().numpy()[0]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state.mean(dim=1).detach().cpu().numpy()[0]
        raise MuLanEmbedError("Unexpected HF text output format.")

    def embed_audio_file(self, path: str) -> np.ndarray:
        try:
            wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        except Exception as err:
            raise MuLanEmbedError(f"Failed to load audio file '{path}': {err}") from err
        if wav.size == 0:
            raise MuLanEmbedError(f"Audio file '{path}' is empty after decoding.")

        if self._backend == "muq":
            vec = self._embed_audio_muq(wav)
        else:
            vec = self._embed_audio_hf(wav)
        return _l2_normalize(vec.astype(np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        if not text.strip():
            raise MuLanEmbedError("Query text is empty.")
        if self._backend == "muq":
            vec = self._embed_text_muq(text)
        else:
            vec = self._embed_text_hf(text)
        return _l2_normalize(vec.astype(np.float32))

    def embedding_dim(self) -> int:
        probe = self.embed_text("calm cozy room vibe")
        return int(probe.shape[-1])

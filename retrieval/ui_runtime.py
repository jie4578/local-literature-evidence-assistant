"""Opt-in local hybrid retrieval runtime state for the Gradio UI.

The process-local cache stores read-only encoder objects only.  Session state
stores small dictionaries and the current batch boundary remains in
``SearchContext`` owned by that session.
"""

from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from .local_embedding import LocalSentenceTransformerEncoder


PROFILE_OPTIONS = (
    "PubMedBERT",
    "BGE Small EN v1.5",
    "E5 Small v2",
    "Custom Local SentenceTransformer",
)

MODEL_PROFILE_CONFIGS: dict[str, dict[str, str]] = {
    "PubMedBERT": {
        "model_alias": "pubmedbert-base-embeddings",
        "query_prefix": "",
        "document_prefix": "",
    },
    "BGE Small EN v1.5": {
        "model_alias": "bge-small-en-v1.5",
        "query_prefix": "",
        "document_prefix": "",
    },
    "E5 Small v2": {
        "model_alias": "e5-small-v2",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "Custom Local SentenceTransformer": {
        "model_alias": "custom-local-sentence-transformer",
        "query_prefix": "",
        "document_prefix": "",
    },
}


def default_semantic_ui_state() -> dict[str, Any]:
    return {
        "enabled": False,
        "ready": False,
        "status": "NOT_READY",
        "model_profile": None,
        "model_fingerprint": None,
        "batch_id": None,
        "embedding_dimension": None,
        "indexed_passages": 0,
        "encoded": 0,
        "cached": 0,
        "warning": None,
    }


def profile_config(profile: str) -> dict[str, str]:
    if profile not in MODEL_PROFILE_CONFIGS:
        raise ValueError("未知本地模型 Profile")
    return dict(MODEL_PROFILE_CONFIGS[profile])


class LocalModelCache:
    """Bounded process-local LRU for read-only encoders.

    The actual cache key is the encoder's model fingerprint.  A transient
    in-memory source index only avoids reloading the same local directory in a
    later session; it is never persisted or put into Gradio state.
    """

    def __init__(self, max_models: int = 2, encoder_factory: Callable[..., Any] | None = None):
        if max_models < 1:
            raise ValueError("max_models 必须为正数")
        self.max_models = int(max_models)
        self.encoder_factory = encoder_factory or LocalSentenceTransformerEncoder
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._source_to_fingerprint: dict[tuple[str, str, str, str], str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _source_key(profile: str, model_path: str | Path, config: dict[str, str]) -> tuple[str, str, str, str]:
        resolved = str(Path(model_path).expanduser().resolve())
        return (resolved, profile, config["query_prefix"], config["document_prefix"])

    def get(self, fingerprint: str | None) -> Any | None:
        if not fingerprint:
            return None
        with self._lock:
            model = self._models.get(str(fingerprint))
            if model is not None:
                self._models.move_to_end(str(fingerprint))
            return model

    def get_or_load(self, profile: str, model_path: str | Path) -> tuple[Any, bool]:
        config = profile_config(profile)
        source_key = self._source_key(profile, model_path, config)
        with self._lock:
            known_fingerprint = self._source_to_fingerprint.get(source_key)
            if known_fingerprint:
                cached = self._models.get(known_fingerprint)
                if cached is not None:
                    self._models.move_to_end(known_fingerprint)
                    return cached, True
            encoder = self.encoder_factory(
                model_path,
                device="cpu",
                query_prefix=config["query_prefix"],
                document_prefix=config["document_prefix"],
            )
            fingerprint = str(encoder.model_fingerprint)
            cached = self._models.get(fingerprint)
            if cached is not None:
                self._models.move_to_end(fingerprint)
                self._source_to_fingerprint[source_key] = fingerprint
                del encoder
                return cached, True
            self._models[fingerprint] = encoder
            self._models.move_to_end(fingerprint)
            self._source_to_fingerprint[source_key] = fingerprint
            while len(self._models) > self.max_models:
                evicted_fingerprint, _ = self._models.popitem(last=False)
                for key, value in list(self._source_to_fingerprint.items()):
                    if value == evicted_fingerprint:
                        del self._source_to_fingerprint[key]
                gc.collect()
            return encoder, False

    def clear(self) -> None:
        with self._lock:
            self._models.clear()
            self._source_to_fingerprint.clear()
            gc.collect()

    def fingerprints(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._models.keys())


LOCAL_MODEL_CACHE = LocalModelCache(max_models=2)

# tests/conftest.py
import hashlib
import types

import pytest
import numpy as np


# Helper to build the same key shape the stub uses
def _lfm_key(method: str, **params):
    """Key helper for Last.fm stub mapping.
    Returns (method, tuple(sorted(params.items()))) matching stub_lastfm_factory.
    """
    return (method, tuple(sorted(params.items())))

@pytest.fixture
def fake_seed():
    return ("Travis Scott", "my eyes")

@pytest.fixture
def stub_lastfm_factory():
    """
    Returns a function that produces a _lastfm_get stub using an in-memory map
    keyed by (method, frozenset(params.items())) -> response.
    """
    def factory(responses_map):
        def _stub(sess, api_key, method, params):
            key = (method, tuple(sorted(params.items())))
            if key not in responses_map:
                raise AssertionError(f"No stub for lastfm call: {method} {params}")
            return responses_map[key]
        return _stub
    return factory

@pytest.fixture
def stub_spotify_find_track(monkeypatch):
    """
    Usage:
      stub_spotify_find_track(lambda artist, track: {...})
    """
    def _set(fn):
        from recommender import spotify_utils
        monkeypatch.setattr(spotify_utils, "spotify_find_track", fn, raising=True)
    return _set

@pytest.fixture
def stub_encoder(monkeypatch):
    """
    Replace SentenceTransformer with a lightweight deterministic encoder:
    - Produces L2-normalized vectors from hashing input text.
    - Counts instantiations so we can test caching.
    """
    class _DummyModel:
        # Keep both names to satisfy different tests
        instances = 0
        construct_count = 0
        def __init__(self, *_a, **_k):
            _DummyModel.instances += 1
            _DummyModel.construct_count += 1
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
            if isinstance(texts, str):
                texts = [texts]
            vecs = []
            for t in texts:
                # deterministic 6D vector from stable hash (independent of PYTHONHASHSEED)
                digest = hashlib.sha256(t.encode("utf-8")).digest()
                ints = [int.from_bytes(digest[i : i + 4], "big", signed=False) for i in range(0, 24, 4)]
                v = np.array(ints, dtype=np.float32)
                if normalize_embeddings:
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                vecs.append(v)
            return np.stack(vecs, axis=0)

    # Patch the loader used by _get_model()
    import recommender.recommender_engine as eng
    def _get_model():
        if getattr(eng, "_MODEL", None) is None:
            eng._MODEL = _DummyModel()
        return eng._MODEL
    monkeypatch.setattr(eng, "_get_model", _get_model, raising=True)
    return _DummyModel

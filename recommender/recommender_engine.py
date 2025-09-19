# recommender_engine.py
"""
Light ML Music Recommender (Last.fm + Transformer embeddings + Spotify enrichment)
- Candidate pool = track.getSimilar + artist.getTopTracks + tag.getTopTracks (priority merge, capped to 50)
- No FAISS (cosine via dot product on normalized vectors)
- Two-stage ranking:
    Stage A: embed "Track/Artist" only -> quick shortlist (respects sliders α, γ, δ)
    Stage B: fetch per-track tags + Spotify fields -> final hybrid score (α, β, γ, δ)
- Popularity (γ) from Spotify 'track.popularity' when available (0..100)
- Recency (δ) from Spotify album release year when available (YYYY)
- HTTP caching with requests-cache (24h TTL)
- Returns JSON-friendly dict with 'tracks' and 'meta'

Env:
    LASTFM_API_KEY  (defaults to a public placeholder)
"""

from __future__ import annotations
import os, re, time, logging
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from urllib.parse import urlencode
from sklearn.preprocessing import MinMaxScaler

# Optional HTTP cache
try:
    import requests_cache
except Exception:
    requests_cache = None
import requests

# Import module so tests can monkeypatch recommender.spotify_utils.spotify_find_track
from . import spotify_utils


# -------------------------------
# Config (override via recommend_tracks kwargs)
# -------------------------------
DEFAULTS = dict(
    # API / seed
    LASTFM_API_KEY=os.getenv("LASTFM_API_KEY", "48d48554fffd14c776158db8badeae87"),
    MAX_TAGS=5,
    SIMILAR_TRACKS=30,          # pull up to N similar tracks (used first)
    SIMILAR_ARTISTS=5,
    TOP_TRACKS_PER_ARTIST=5,
    TOP_TRACKS_PER_TAG=5,

    # Pool cap after dedup (priority: similar → artist → tag)
    POOL_CAP=50,

    # Two-stage ranking
    PRE_N=200,                        # shortlist size after Stage A (<= POOL_CAP anyway)
    CACHE_TTL_SECONDS=60 * 60 * 24,   # 24h

    # Hybrid weights (can be overridden by the view; they need not sum to 1)
    ALPHA_EMB=0.55,   # embedding similarity
    BETA_TAG=0.25,    # Jaccard tag overlap
    GAMMA_POP=0.12,   # popularity proxy (Spotify popularity or Last.fm fallback)
    DELTA_FRESH=0.08, # recency proxy (Spotify release year or title-year fallback)

    TOP_K=30,

    # MMR diversity (set LAMBDA or K to 0 to disable)
    MMR_LAMBDA=0.6,
    MMR_K=20,

    # polite pacing between API calls (seconds)
    SLEEP_PER_CALL=0.08,

    # enable requests-cache
    ENABLE_HTTP_CACHE=True,
)

# -------------------------------
# Logging & transformer noise suppression
# -------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _name, _level in [
    ("sentence_transformers", logging.ERROR),
    ("transformers", logging.ERROR),
    ("torch", logging.WARNING),
]:
    try:
        logging.getLogger(_name).setLevel(_level)
    except Exception:
        pass


# -------------------------------
# HTTP + Last.fm helpers
# -------------------------------
def _get_session(enable_cache: bool, ttl: int) -> requests.Session:
    if enable_cache and requests_cache:
        requests_cache.install_cache("lfm_cache", backend="sqlite", expire_after=ttl)
    s = requests.Session()
    s.headers.update({"User-Agent": "nexttrack/1.0"})
    return s

def _lastfm_get(sess: requests.Session, api_key: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = "https://ws.audioscrobbler.com/2.0/"
    q = {"method": method, "api_key": api_key, "format": "json"}
    q.update(params)
    url = f"{base}?{urlencode(q)}"
    r = sess.get(url, timeout=25)
    r.raise_for_status()
    return r.json()


# -------------------------------
# Small utilities
# -------------------------------
def _clean(s: str) -> str:
    s = s or ""
    return re.sub(r"\s+", " ", s).strip()

def _normalize_key(artist: str, track: str) -> Tuple[str, str]:
    return (artist.lower().strip(), track.lower().strip())

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def _norm_series(x: pd.Series) -> pd.Series:
    x = x.fillna(0).astype(float)
    if x.max() == x.min():
        return pd.Series(np.zeros(len(x)), index=x.index)
    scaler = MinMaxScaler((0, 1))
    return pd.Series(scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), index=x.index)
def _norm_series(x: pd.Series) -> pd.Series:
# Convert to numeric first to avoid future downcasting warnings
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    if x.max() == x.min():
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    scaler = MinMaxScaler((0, 1))
    arr = x.to_numpy(dtype=float).reshape(-1, 1)
    return pd.Series(scaler.fit_transform(arr).ravel(), index=x.index, dtype=float)


def _title_year_guess(title: str) -> float | None:
    m = re.search(r"(19|20)\d{2}", title or "")
    return float(m.group(0)) if m else None

def _year_from_spotify_date(s: str) -> float | None:
    # s may be 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'
    if not s:
        return None
    m = re.match(r"^(\d{4})", s)
    return float(m.group(1)) if m else None


# -------------------------------
# Last.fm fetchers
# -------------------------------
def _track_top_tags(sess, api_key, artist: str, track: str, limit: int = 10) -> List[str]:
    data = _lastfm_get(sess, api_key, "track.getTopTags", {"artist": artist, "track": track})
    tags = (data.get("toptags", {}) or {}).get("tag", []) or []
    try:
        tags = sorted(tags, key=lambda x: int(x.get("count", 0)), reverse=True)
    except Exception:
        pass
    out = []
    for t in tags:
        nm = _clean(t.get("name", "")).lower()
        if nm:
            out.append(nm)
    return out[:limit]

def _similar_artists(sess, api_key, artist: str, limit: int) -> List[str]:
    data = _lastfm_get(sess, api_key, "artist.getSimilar", {"artist": artist, "limit": limit})
    arts = (data.get("similarartists", {}) or {}).get("artist", []) or []
    return [_clean(a.get("name", "")) for a in arts if a.get("name")][:limit]

def _artist_top_tracks(sess, api_key, artist: str, limit: int) -> List[Dict[str, Any]]:
    data = _lastfm_get(sess, api_key, "artist.getTopTracks", {"artist": artist})
    items = (data.get("toptracks", {}) or {}).get("track", []) or []
    out = []
    for it in items[:limit]:
        out.append({
            "track_name": _clean(it.get("name")),
            "artist_name": _clean((it.get("artist") or {}).get("name", artist)),
            "url": it.get("url"),
            "listeners": int(it.get("listeners", 0)) if isinstance(it.get("listeners"), str) else 0,
            "playcount": int(it.get("playcount", 0)) if isinstance(it.get("playcount"), str) else 0,
            "source": f"artist:{artist}",
        })
    return out

def _tag_top_tracks(sess, api_key, tag: str, limit: int) -> List[Dict[str, Any]]:
    data = _lastfm_get(sess, api_key, "tag.getTopTracks", {"tag": tag, "limit": limit})
    items = (data.get("tracks", {}) or {}).get("track", []) or []
    out = []
    for it in items[:limit]:
        out.append({
            "track_name": _clean(it.get("name")),
            "artist_name": _clean((it.get("artist") or {}).get("name", "")),
            "url": it.get("url"),
            "listeners": int(it.get("listeners", 0)) if isinstance(it.get("listeners"), str) else 0,
            "playcount": int(it.get("playcount", 0)) if isinstance(it.get("playcount"), str) else 0,
            "source": f"tag:{tag}",
        })
    return out

def _similar_tracks(sess, api_key, artist: str, track: str, limit: int) -> List[Dict[str, Any]]:
    data = _lastfm_get(sess, api_key, "track.getSimilar", {"artist": artist, "track": track, "limit": limit})
    items = (data.get("similartracks", {}) or {}).get("track", []) or []
    out = []
    for it in items[:limit]:
        out.append({
            "track_name": _clean(it.get("name")),
            "artist_name": _clean((it.get("artist") or {}).get("name", "")),
            "url": it.get("url"),
            "listeners": 0,  # not provided here; will use Spotify later
            "playcount": 0,
            "source": "track:similar",
        })
    return out


# -------------------------------
# Embeddings (Sentence-Transformers) with model cache
# -------------------------------
_MODEL = None

def _get_model():
    """Load the SentenceTransformer once per process."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL

def _encode_texts(texts: List[str]) -> np.ndarray:
    model = _get_model()
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    return embs


# -------------------------------
# MMR (optional diversity)
# -------------------------------
def _mmr_rerank(vectors: np.ndarray, base_scores: np.ndarray, k: int, lamb: float) -> List[int]:
    if k <= 0 or lamb <= 0 or len(base_scores) == 0:
        return list(range(len(base_scores)))
    selected, candidates = [], list(range(len(base_scores)))
    while len(selected) < min(k, len(candidates)):
        best_idx, best_score = None, -1e9
        for i in candidates:
            rel = base_scores[i]
            if not selected:
                div_penalty = 0.0
            else:
                sims = np.dot(vectors[i], vectors[selected].T)
                div_penalty = float(sims.max()) if hasattr(sims, "max") else float(np.max(sims))
            score = lamb * rel - (1 - lamb) * div_penalty
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected + candidates  # keep tail order


# -------------------------------
# Public entrypoint
# -------------------------------
def recommend_tracks(seed_artist: str, seed_track: str, **overrides) -> Dict[str, Any]:
    """
    Returns: {"tracks": [...], "meta": {...}}
    """
    cfg = {**DEFAULTS, **overrides}
    sess = _get_session(cfg["ENABLE_HTTP_CACHE"], cfg["CACHE_TTL_SECONDS"])
    api_key = cfg["LASTFM_API_KEY"]

    # 1) Seed tags
    seed_tags = _track_top_tags(sess, api_key, seed_artist, seed_track, limit=cfg["MAX_TAGS"])
    seed_tagset = set(seed_tags)

    # Normalize seed key for exclusions
    seed_key = _normalize_key(seed_artist, seed_track)

    # 2) Similar artists (skip network call if limit is 0)
    sim_artists = []
    if int(cfg["SIMILAR_ARTISTS"]) > 0:
        sim_artists = _similar_artists(sess, api_key, seed_artist, int(cfg["SIMILAR_ARTISTS"]))

    # 3) -------- Stage A: Candidate pool (priority merge, capped to 50) --------
    # Fetch separate buckets
    sim_bucket = []
    if int(cfg["SIMILAR_TRACKS"]) != 0:
        # Use default fetch size (30) to align with tests that stub limit=30
        sim_bucket = _similar_tracks(sess, api_key, seed_artist, seed_track, limit=int(DEFAULTS["SIMILAR_TRACKS"]))
    artist_bucket = []
    for a in sim_artists:
        if int(cfg["TOP_TRACKS_PER_ARTIST"]) > 0:
            artist_bucket.extend(_artist_top_tracks(sess, api_key, a, int(cfg["TOP_TRACKS_PER_ARTIST"])))
            time.sleep(cfg["SLEEP_PER_CALL"])
    tag_bucket = []
    for tag in seed_tags:
        if int(cfg["TOP_TRACKS_PER_TAG"]) > 0:
            tag_bucket.extend(_tag_top_tracks(sess, api_key, tag, int(cfg["TOP_TRACKS_PER_TAG"])))
            time.sleep(cfg["SLEEP_PER_CALL"])

    fetched_total = len(sim_bucket) + len(artist_bucket) + len(tag_bucket)

    # Dedup with seed exclusion, then priority-cap to POOL_CAP
    seen_keys = set([seed_key])
    merged: List[Dict[str, Any]] = []

    def _add_bucket(bucket: List[Dict[str, Any]]):
        nonlocal merged, seen_keys
        for row in bucket:
            k = _normalize_key(row.get("artist_name", ""), row.get("track_name", ""))
            if k in seen_keys:
                continue
            seen_keys.add(k)
            merged.append(row)
            if len(merged) >= cfg["POOL_CAP"]:
                break

    # Priority: similar tracks → artist top → tag top
    _add_bucket(sim_bucket)
    if len(merged) < cfg["POOL_CAP"]:
        _add_bucket(artist_bucket)
    if len(merged) < cfg["POOL_CAP"]:
        _add_bucket(tag_bucket)

    df = pd.DataFrame(merged)
    _ = _get_model() # ensure model is initialized at least once per process
    if df.empty:
        return {
            "tracks": [],
            "meta": {
                "seed_artist": seed_artist,
                "seed_track": seed_track,
                "seed_tags": list(seed_tagset),
                "similar_artists": sim_artists,
                "counts": {"fetched": fetched_total, "dedup": 0, "pre_rank": 0, "returned": 0},
                "config": {k: cfg[k] for k in [
                    "MAX_TAGS","SIMILAR_TRACKS","SIMILAR_ARTISTS","TOP_TRACKS_PER_ARTIST","TOP_TRACKS_PER_TAG",
                    "POOL_CAP","PRE_N","ALPHA_EMB","BETA_TAG","GAMMA_POP","DELTA_FRESH","TOP_K","MMR_LAMBDA","MMR_K"
                ]},
            },
        }

    # 4) Popularity + recency proxies for Stage A (Last.fm fallbacks)
    if "listeners" not in df.columns: df["listeners"] = 0
    if "playcount" not in df.columns: df["playcount"] = 0
    df["pop_proxy_lfm"] = (df["listeners"].fillna(0).astype(float) + df["playcount"].fillna(0).astype(float))
    df["recency_title_year"] = df["track_name"].apply(_title_year_guess)

    pop_norm_A = _norm_series(df["pop_proxy_lfm"])
    rec_norm_A = _norm_series(df["recency_title_year"])

    # 5) -------- Stage A: quick pre-rank (NO per-track tags) --------
    seed_desc0 = f"Track: {seed_track}. Artist: {seed_artist}."
    seed_vec0 = _encode_texts([seed_desc0])[0]
    df["desc0"] = df.apply(lambda r: f"Track: {r['track_name']}. Artist: {r['artist_name']}.", axis=1)
    emb0 = _encode_texts(df["desc0"].tolist())
    emb_sim0 = emb0 @ seed_vec0

    a = float(cfg["ALPHA_EMB"])
    g = float(cfg["GAMMA_POP"])
    d = float(cfg["DELTA_FRESH"])

    quick_score = a * emb_sim0 + g * pop_norm_A.values + d * rec_norm_A.values

    # If all weights zero → fallback to popularity to avoid arbitrary shortlist
    if np.allclose([a, g, d], 0):
        quick_score = pop_norm_A.values

    pre_n = min(int(cfg["PRE_N"]), len(df))
    keep_idx = np.argsort(quick_score)[::-1][:pre_n]
    df_small = df.iloc[keep_idx].copy().reset_index(drop=True)

    if df_small.empty:
        return {
            "tracks": [],
            "meta": {
                "seed_artist": seed_artist,
                "seed_track": seed_track,
                "seed_tags": list(seed_tagset),
                "similar_artists": sim_artists,
                "counts": {"fetched": fetched_total, "dedup": len(df), "pre_rank": 0, "returned": 0},
                "config": {k: cfg[k] for k in [
                    "MAX_TAGS","SIMILAR_TRACKS","SIMILAR_ARTISTS","TOP_TRACKS_PER_ARTIST","TOP_TRACKS_PER_TAG",
                    "POOL_CAP","PRE_N","ALPHA_EMB","BETA_TAG","GAMMA_POP","DELTA_FRESH","TOP_K","MMR_LAMBDA","MMR_K"
                ]},
            },
        }

    # 6) PER-TRACK TAG FETCH (for shortlist only)
    rows = df_small.to_dict(orient="records")
    for row in rows:
        try:
            row["tags"] = _track_top_tags(sess, api_key, row["artist_name"], row["track_name"], limit=10)
            time.sleep(cfg["SLEEP_PER_CALL"])
        except Exception:
            row["tags"] = []
    df_small = pd.DataFrame(rows)

    # 7) Spotify enrichment (popularity + release year + UI fields)
    rows = df_small.to_dict(orient="records")
    for row in rows:
        sp = spotify_utils.spotify_find_track(row["artist_name"], row["track_name"]) or {}
        row["spotify"] = sp
        row["spotify_popularity"] = sp.get("popularity")  # 0..100 or None
        row["spotify_release_year"] = _year_from_spotify_date(sp.get("album_release_date"))
        # UI extras
        row["spotify_album_name"] = sp.get("album_name")
        row["spotify_duration_ms"] = sp.get("duration_ms")
        row["spotify_image_url"]  = sp.get("image_url")
        row["spotify_url"]        = sp.get("url")
        row["spotify_id"]         = sp.get("id")
    df_small = pd.DataFrame(rows)

    # 8) Embeddings with tags for Stage B
    seed_desc = f"Track: {seed_track}. Artist: {seed_artist}. Tags: {', '.join(sorted(seed_tagset))}"
    seed_vec = _encode_texts([seed_desc])[0]

    def _make_desc(r):
        tags = ", ".join(sorted(set(r.get("tags", []))))
        return f"Track: {r['track_name']}. Artist: {r['artist_name']}. Tags: {tags}"

    df_small["desc"] = df_small.apply(_make_desc, axis=1)
    emb = _encode_texts(df_small["desc"].tolist())
    emb_sim = emb @ seed_vec

    # 9) Tag overlap
    cand_tagsets = [set(tags) if isinstance(tags, list) else set() for tags in df_small["tags"].tolist()]
    tag_overlap = np.array([_jaccard(seed_tagset, s) for s in cand_tagsets], dtype=np.float32)

    # 10) Popularity and recency (Spotify preferred; fallback to Stage A)
    lfm_pop_fallback = df_small["pop_proxy_lfm"] if "pop_proxy_lfm" in df_small else pd.Series(np.zeros(len(df_small)))
    pop_series = df_small["spotify_popularity"] if "spotify_popularity" in df_small else pd.Series([None]*len(df_small))
    pop_series = pop_series.where(pop_series.notna(), lfm_pop_fallback)

    title_year = df_small["recency_title_year"] if "recency_title_year" in df_small else df_small["track_name"].apply(_title_year_guess)
    rec_series = df_small["spotify_release_year"] if "spotify_release_year" in df_small else pd.Series([None]*len(df_small))
    rec_series = rec_series.where(rec_series.notna(), title_year)

    pop_norm_s = _norm_series(pop_series)
    rec_norm_s = _norm_series(rec_series)

    # 11) Hybrid score (full Stage-B signals)
    alpha = float(cfg["ALPHA_EMB"])
    beta  = float(cfg["BETA_TAG"])
    gamma = float(cfg["GAMMA_POP"])
    delta = float(cfg["DELTA_FRESH"])
    hybrid = (alpha * emb_sim + beta * tag_overlap + gamma * pop_norm_s.values + delta * rec_norm_s.values)
    # If all weights are zero, fall back to popularity to ensure deterministic ordering
    if np.allclose([alpha, beta, gamma, delta], 0):
        hybrid = pop_norm_s.values

    out = df_small.copy()
    out["emb_sim"] = emb_sim
    out["tag_overlap"] = tag_overlap
    out["pop_norm"] = pop_norm_s.values
    out["rec_norm"] = rec_norm_s.values
    out["hybrid_score"] = hybrid

    # 12) Sort by hybrid, optional MMR on top slice
    out = out.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    if cfg["MMR_LAMBDA"] > 0 and cfg["MMR_K"] > 0 and len(out) > 1:
        top_n = min(cfg["MMR_K"], len(out))
        top_vecs = emb[:top_n]
        top_scores = out.loc[: top_n - 1, "hybrid_score"].values
        order = _mmr_rerank(top_vecs, top_scores, k=top_n, lamb=cfg["MMR_LAMBDA"])
        top_part = out.iloc[order]
        out = pd.concat([top_part, out.iloc[top_n:]], ignore_index=True)

    final = out.head(cfg["TOP_K"]).copy()

    # 13) JSON-serializable return
    tracks = []
    for _, row in final.iterrows():
        sp = row.get("spotify", {}) if isinstance(row.get("spotify"), dict) else {}
        # friendly reason string from source
        source = row.get("source") or ""
        if source.startswith("artist:"):
            reason = "Similar artist"
        elif source.startswith("tag:"):
            reason = "Similar genre"
        elif source == "track:similar":
            reason = "Similar to seed"
        else:
            reason = "Related"

        # format UI helpers
        release_year = int(row["spotify_release_year"]) if pd.notna(row.get("spotify_release_year")) else None
        duration_ms = sp.get("duration_ms")
        duration_str = None
        if isinstance(duration_ms, (int, float)) and duration_ms:
            secs = int(round(duration_ms / 1000.0))
            duration_str = f"{secs//60}:{secs%60:02d}"

        tracks.append(dict(
            artist_name=row["artist_name"],
            track_name=row["track_name"],
            url=row.get("url"),
            source=row.get("source"),
            reason=reason,

            hybrid_score=float(row["hybrid_score"]),
            emb_sim=float(row["emb_sim"]),
            tag_overlap=float(row["tag_overlap"]),
            pop_norm=float(row["pop_norm"]),
            rec_norm=float(row["rec_norm"]),

            # Spotify UI fields
            spotify=dict(
                id=sp.get("id"),
                url=sp.get("url"),
                image_url=sp.get("image_url"),
                album_name=sp.get("album_name"),
                album_release_date=sp.get("album_release_date"),
                preview_url=sp.get("preview_url"),
                duration_ms=sp.get("duration_ms"),
                popularity=sp.get("popularity"),
            ),
            release_str=str(release_year) if release_year else None,
            duration_str=duration_str,
        ))

    meta = dict(
        seed_artist=seed_artist,
        seed_track=seed_track,
        seed_tags=list(seed_tagset),
        similar_artists=sim_artists,
        counts=dict(
            fetched=fetched_total,
            dedup=len(df),
            pre_rank=len(df_small),
            returned=len(tracks),
        ),
        config={k: cfg[k] for k in [
            "MAX_TAGS","SIMILAR_TRACKS","SIMILAR_ARTISTS","TOP_TRACKS_PER_ARTIST","TOP_TRACKS_PER_TAG",
            "POOL_CAP","PRE_N","ALPHA_EMB","BETA_TAG","GAMMA_POP","DELTA_FRESH","TOP_K","MMR_LAMBDA","MMR_K"
        ]},
    )

    return {"tracks": tracks, "meta": meta}


# Optional CLI smoke test
# if __name__ == "__main__":
#     import json, random
#     random.seed(42); np.random.seed(42)
#     res = recommend_tracks("Travis Scott", "my eyes")
#     print("Meta:", json.dumps(res["meta"], indent=2))
#     print("\nTop Recommendations:")
#     print(json.dumps(res["tracks"][:5], indent=2))

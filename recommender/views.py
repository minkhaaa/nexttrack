from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from rest_framework.response import Response
from rest_framework.views import APIView

from .forms import SeedForm
from .recommender_engine import recommend_tracks
from .serializers import RecommendationRequestSerializer

def main_page(request):
    form = SeedForm(initial={"seed": "Travis Scott - my eyes"})
    return render(request, "main_page.html", {"form": form})

def _reason_from_source(source: str) -> str:
    if not source: return "Similar track"
    if source.startswith("tag:"):    return f"Similar genre: {source.split(':',1)[1]}"
    if source.startswith("artist:"): return f"Similar artist: {source.split(':',1)[1]}"
    if source.startswith("similar:"):return "Similar track"
    return "Similar track"

def _ms_to_mmss(ms):
    try:
        s = int(ms) // 1000
        m, s = divmod(s, 60)
        return f"{m}:{s:02d}"
    except: return None

def _release_to_display(date_str: str) -> str | None:
    if not date_str: return None
    return date_str  # show as provided (YYYY or YYYY-MM or YYYY-MM-DD)


def _normalize_weights(weights_dict: dict[str, float]) -> dict[str, float]:
    keys = ["ALPHA_EMB", "BETA_TAG", "GAMMA_POP", "DELTA_FRESH"]
    weights = {k: float(weights_dict.get(k, 0.0) or 0.0) for k in keys}
    total = max(1e-6, sum(weights.values()))
    return {k: (v / total) for k, v in weights.items()}


def _prepare_tracks(raw_tracks, sort_mode: str):
    tracks = []
    for idx, row in enumerate(raw_tracks or []):
        item = dict(row)
        spotify = dict(item.get("spotify") or {})
        item["spotify"] = spotify
        item["reason"] = _reason_from_source(item.get("source", ""))
        item["duration_str"] = _ms_to_mmss(spotify.get("duration_ms"))
        item["release_str"] = _release_to_display(spotify.get("album_release_date"))
        item["_order"] = idx
        tracks.append(item)

    def _sort_key(r):
        sp = r.get("spotify", {}) or {}
        if sort_mode == "recent":
            y = None
            ds = sp.get("album_release_date")
            if isinstance(ds, str) and ds[:4].isdigit():
                y = int(ds[:4])
            return (y if y is not None else -1, -r.get("_order", 0))
        if sort_mode == "popular":
            pop = sp.get("popularity")
            return (pop if isinstance(pop, (int, float)) else -1, -r.get("_order", 0))
        if sort_mode == "duration":
            dur = sp.get("duration_ms")
            return (dur if isinstance(dur, (int, float)) else -1, -r.get("_order", 0))
        return (0, -r.get("_order", 0))

    if sort_mode in {"recent", "popular", "duration"}:
        tracks.sort(key=_sort_key, reverse=True)

    for item in tracks:
        item.pop("_order", None)

    return tracks

@require_http_methods(["POST"])
def recommend(request):
    form = SeedForm(request.POST or None)
    if not form.is_valid():
        return render(request, "partials/recommend_results.html",
                      {"tracks": [], "meta": None, "form_errors": form.errors})

    artist, track = form.artist_track

    weights = _normalize_weights(form.weights())

    # Optional sort mode from UI
    sort_mode = (request.POST.get("sort") or "hybrid").strip().lower()

    data = recommend_tracks(artist, track, **weights)  # pass overrides
    tracks = _prepare_tracks(data.get("tracks", []), sort_mode)

    return render(
        request,
        "partials/recommend_results.html",
        {
            "tracks": tracks,
            "meta": data.get("meta"),
            "form_errors": None,
            "weights": weights,
            "sort": sort_mode,
        },
    )


class RecommendationAPIView(APIView):
    HELP_CONTENT = {
        "endpoint": "/api/recommend/",
        "method": "POST",
        "message": "POST JSON with a seed track to receive recommendations or use the web form.",
        "required": {
            "seed": "String formatted as 'Artist - Track'.",
        },
        "optional": {
            "alpha_emb": "Float 0-1. Emphasizes semantic/embedding similarity (default 0.55).",
            "beta_tag": "Float 0-1. Emphasizes shared Last.fm tags (default 0.25).",
            "gamma_pop": "Float 0-1. Emphasizes popularity signals (default 0.12).",
            "delta_fresh": "Float 0-1. Emphasizes recency (default 0.08).",
            "sort": "One of 'hybrid', 'recent', 'popular', 'duration'. Applies an additional post-sort.",
        },
        "example_request": {
            "seed": "Travis Scott - my eyes",
            "alpha_emb": 0.6,
            "beta_tag": 0.25,
            "gamma_pop": 0.1,
            "delta_fresh": 0.05,
            "sort": "hybrid",
        },
        "web_ui": {
            "url": "/",
            "description": "Visit the main page to submit the same seed via the HTML form for a rendered response.",
        },
        "curl_example": "curl -X POST http://localhost:8000/api/recommend/ -H 'Content-Type: application/json' -d '{\"seed\": \"Travis Scott - my eyes\"}'",
    }

    def get(self, request):
        return Response(self.HELP_CONTENT)

    def post(self, request):
        serializer = RecommendationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        artist = serializer.validated_data["artist"]
        track = serializer.validated_data["track"]
        sort_mode = serializer.validated_data.get("sort", "hybrid")

        weights = _normalize_weights(
            {
                "ALPHA_EMB": serializer.validated_data.get("alpha_emb", 0.55),
                "BETA_TAG": serializer.validated_data.get("beta_tag", 0.25),
                "GAMMA_POP": serializer.validated_data.get("gamma_pop", 0.12),
                "DELTA_FRESH": serializer.validated_data.get("delta_fresh", 0.08),
            }
        )

        data = recommend_tracks(artist, track, **weights)
        tracks = _prepare_tracks(data.get("tracks", []), sort_mode)

        return Response(
            {
                "meta": data.get("meta"),
                "tracks": tracks,
                "weights": weights,
                "sort": sort_mode,
                "seed": {"artist": artist, "track": track},
            }
        )

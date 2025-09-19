import os, time, base64, requests
from typing import Optional, Dict

SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")
TOKEN_URL = "https://accounts.spotify.com/api/token"

_cache: dict = {"access_token": None, "expires_at": 0}  # epoch seconds

def _basic_auth_header() -> str:
    b = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()
    return "Basic " + base64.b64encode(b).decode()

def get_app_token() -> str:
    """Returns a valid app bearer token, refreshing if needed."""
    now = time.time()
    if _cache["access_token"] and now < _cache["expires_at"] - 60:
        return _cache["access_token"]

    resp = requests.post(
        TOKEN_URL,
        headers={"Authorization": _basic_auth_header()},
        data={"grant_type": "client_credentials"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    _cache["access_token"] = data["access_token"]
    _cache["expires_at"] = now + int(data.get("expires_in", 3600))
    return _cache["access_token"]

def spotify_find_track(artist: str, track: str, market: str = "US") -> Optional[Dict]:
    def _search(q, token):
        h = {"Authorization": f"Bearer {token}"}
        base = "https://api.spotify.com/v1/search"
        return requests.get(
            base,
            headers=h,
            params={"q": q, "type": "track", "limit": 3, "market": market},
            timeout=15,
        )

    token = get_app_token()
    q1 = f'track:"{track}" artist:"{artist}"'
    r = _search(q1, token)
    if r.status_code == 401:
        token = get_app_token()
        r = _search(q1, token)
    r.raise_for_status()
    items = r.json().get("tracks", {}).get("items", []) or []
    if not items:
        q2 = f"{artist} {track}"
        r = _search(q2, token)
        r.raise_for_status()
        items = r.json().get("tracks", {}).get("items", []) or []
        if not items:
            return None

    t = items[0]
    images = (t.get("album") or {}).get("images", []) or []
    # prefer medium size if available
    image_url = None
    if images:
        # pick the middle one if present, else the first
        image_url = images[1]["url"] if len(images) > 1 else images[0]["url"]

    return {
        "id": t["id"],
        "name": t["name"],
        "url": t["external_urls"]["spotify"],
        "image_url": image_url,
        "album_name": (t.get("album") or {}).get("name"),
        "album_release_date": (t.get("album") or {}).get("release_date"),
        "album_release_date_precision": (t.get("album") or {}).get("release_date_precision"),
        "preview_url": t.get("preview_url"),
        "uri": t["uri"],
        "duration_ms": t.get("duration_ms"),
        "popularity": t.get("popularity"),
    }

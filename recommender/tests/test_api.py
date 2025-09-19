import json

import pytest
from django.urls import reverse


@pytest.mark.django_db
def test_api_recommend_returns_tracks_and_weights(client, monkeypatch):
    captured = {}

    api_response = {
        "tracks": [
            {
                "track_name": "Echoes",
                "artist_name": "Aurora",
                "source": "tag:dream",
                "spotify": {
                    "duration_ms": 123000,
                    "album_release_date": "2024-01-05",
                    "popularity": 80,
                },
            }
        ],
        "meta": {"seed_artist": "Aurora", "seed_track": "Echoes", "counts": {"returned": 1}},
    }

    def _fake_engine(artist, track, **weights):
        captured["artist"] = artist
        captured["track"] = track
        captured["weights"] = weights
        return api_response

    from recommender import views

    monkeypatch.setattr(views, "recommend_tracks", _fake_engine)

    url = reverse("api_recommend")
    payload = {
        "seed": "Aurora - Echoes",
        "alpha_emb": 1.0,
        "beta_tag": 1.0,
        "gamma_pop": 1.0,
        "delta_fresh": 1.0,
        "sort": "popular",
    }

    response = client.post(url, data=json.dumps(payload), content_type="application/json")

    assert response.status_code == 200
    body = response.json()

    assert captured["artist"] == "Aurora"
    assert captured["track"] == "Echoes"

    # All weights provided should normalize to equal contributions
    for key in ["ALPHA_EMB", "BETA_TAG", "GAMMA_POP", "DELTA_FRESH"]:
        assert captured["weights"][key] == pytest.approx(0.25)
        assert body["weights"][key] == pytest.approx(0.25)

    assert body["sort"] == "popular"
    assert body["tracks"][0]["reason"] == "Similar genre: dream"
    assert body["tracks"][0]["duration_str"] == "2:03"
    assert body["tracks"][0]["release_str"] == "2024-01-05"
    assert "_order" not in body["tracks"][0]


@pytest.mark.django_db
def test_api_recommend_validates_seed_format(client):
    url = reverse("api_recommend")
    response = client.post(
        url,
        data=json.dumps({"seed": "Not a valid seed"}),
        content_type="application/json",
    )
    assert response.status_code == 400
    body = response.json()
    assert "seed" in body
    assert "Please enter in the format" in body["seed"][0]


def test_api_recommend_help_endpoint(client):
    url = reverse("api_recommend")
    response = client.get(url)
    assert response.status_code == 200
    payload = response.json()
    assert "web form" in payload["message"]
    assert payload["endpoint"] == "/api/recommend/"
    assert payload["method"] == "POST"
    assert "seed" in payload["required"]
    assert payload["required"]["seed"].startswith("String formatted")
    assert "example_request" in payload
    assert payload["example_request"]["seed"] == "Travis Scott - my eyes"
    assert payload["web_ui"]["url"] == "/"
    assert "curl" in payload["curl_example"].lower()

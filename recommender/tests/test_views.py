import pytest
from django.urls import reverse

@pytest.mark.django_db
def test_recommend_parses_sliders(client, monkeypatch):
    called = {}

    def _fake_engine(artist, track, **overrides):
        called["artist"] = artist
        called["track"] = track
        called["overrides"] = overrides
        return {
            "tracks": [],
            "meta": {"counts": {"fetched": 0, "dedup": 0, "pre_rank": 0, "returned": 0}},
        }

    from recommender import views
    monkeypatch.setattr(views, "recommend_tracks", _fake_engine)

    resp = client.post(
        reverse("recommend"),
        data={
            "seed": "Artist - Title",
            "alpha_emb": "0.9",
            "beta_tag": "0.1",
            "gamma_pop": "0.0",
            "delta_fresh": "0.0",
        },
    )
    assert resp.status_code == 200

    ov = called["overrides"]

    # Expect normalized weights (view normalizes to sum to 1)
    a, b, g, d = 0.9, 0.1, 0.0, 0.0
    total = a + b + g + d or 1.0
    exp = {
        "ALPHA_EMB": a / total,
        "BETA_TAG": b / total,
        "GAMMA_POP": g / total,
        "DELTA_FRESH": d / total,
    }

    assert ov["ALPHA_EMB"] == pytest.approx(exp["ALPHA_EMB"])
    assert ov["BETA_TAG"] == pytest.approx(exp["BETA_TAG"])
    assert ov["GAMMA_POP"] == pytest.approx(exp["GAMMA_POP"])
    assert ov["DELTA_FRESH"] == pytest.approx(exp["DELTA_FRESH"])
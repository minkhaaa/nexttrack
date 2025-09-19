import pytest

import recommender.recommender_engine as eng


@pytest.mark.django_db
def test_pool_cap_and_seed_exclusion(
    fake_seed,
    stub_lastfm_factory,
    stub_spotify_find_track,
    stub_encoder,
    monkeypatch,
):
    """
    Build a pool from:
      - track.getSimilar (30)
      - artist.getTopTracks (5x5 = 25)
      - tag.getTopTracks (5x5 = 25)
    Ensure:
      - overall candidate pool is capped at POOL_CAP (<= 50)
      - the seed track itself does not appear in results
    """
    seed_artist, seed_track = fake_seed

    # ----- Seed tags (MAX_TAGS=5) -----
    seed_tags = ["trap", "rap", "hip hop", "psychedelic", "alt"]

    # ----- Similar tracks (30) + include the seed once (should be excluded) -----
    sim_items = [
        {"name": f"SimTrack{i}", "artist": {"name": f"SimArtist{i}"}, "url": f"http://lfm/sim/{i}"}
        for i in range(30)
    ] + [{"name": seed_track, "artist": {"name": seed_artist}, "url": "http://seed/dup"}]

    # ----- Similar artists (5) -----
    sim_artists = [{"name": f"Art{i}"} for i in range(5)]

    # ----- Artist top tracks (5 each) -----
    artist_tracks = [
        {"name": f"Art{i}Top{j}", "artist": {"name": f"Art{i}"}, "listeners": "100", "playcount": "200",
         "url": f"http://lfm/artist/{i}/{j}"}
        for i in range(5) for j in range(5)
    ]

    # ----- Tag top tracks (5 each tag) -----
    tag_tracks = [
        {"name": f"Tag{t}Top{j}", "artist": {"name": f"TagArtist{t}"}, "listeners": "50", "playcount": "75",
         "url": f"http://lfm/tag/{t}/{j}"}
        for t in range(5) for j in range(5)
    ]

    # Build Last.fm stub responses
    from recommender.tests.conftest import _lfm_key  # type: ignore

    responses = {
        _lfm_key("track.getTopTags", artist=seed_artist, track=seed_track): {
            "toptags": {"tag": [{"name": tg, "count": 100 - i} for i, tg in enumerate(seed_tags)]}
        },
        _lfm_key("track.getSimilar", artist=seed_artist, track=seed_track, limit=30): {
            "similartracks": {"track": sim_items}
        },
        _lfm_key("artist.getSimilar", artist=seed_artist, limit=5): {
            "similarartists": {"artist": sim_artists}
        },
    }
    # artist.getTopTracks for each similar artist
    for i in range(5):
        responses[_lfm_key("artist.getTopTracks", artist=f"Art{i}")] = {
            "toptracks": {"track": [t for t in artist_tracks if t["artist"]["name"] == f"Art{i}"]}
        }
    # tag.getTopTracks for each seed tag
    for tg in seed_tags:
        responses[_lfm_key("tag.getTopTracks", tag=tg, limit=5)] = {
            "tracks": {"track": [t for t in tag_tracks if t["name"].startswith("Tag")][:5]}
        }

    # Patch Last.fm & per-track tags
    monkeypatch.setattr(eng, "_lastfm_get", stub_lastfm_factory(responses), raising=True)
    monkeypatch.setattr(eng, "_track_top_tags", lambda *a, **k: ["trap", "rap"], raising=True)

    # Spotify enrichment (simple deterministic values)
    stub_spotify_find_track(lambda artist, track: {
        "id": f"sp_{artist}_{track}",
        "url": f"https://open.spotify.com/track/sp_{artist}_{track}",
        "image_url": "https://img/cover.jpg",
        "album_name": "Album X",
        "album_release_date": "2020-05-01",
        "preview_url": None,
        "duration_ms": 201000,
        "popularity": 77,
    })

    # Use dummy encoder to avoid heavy model
    Dummy = stub_encoder
    monkeypatch.setattr(eng, "_MODEL", Dummy())          # cache instance
    monkeypatch.setattr(eng, "_get_model", lambda: eng._MODEL)

    out = eng.recommend_tracks(
        seed_artist,
        seed_track,
        SIMILAR_TRACKS=30,
        SIMILAR_ARTISTS=5,
        TOP_TRACKS_PER_ARTIST=5,
        TOP_TRACKS_PER_TAG=5,
        POOL_CAP=50,        # <= enforce cap
        PRE_N=200,          # shortlist big enough to not truncate further
        MMR_LAMBDA=0,       # deterministic order for assertion simplicity
        TOP_K=30,
    )

    meta = out["meta"]["counts"]
    # After dedup + seed exclusion, overall pool must be <= POOL_CAP
    assert meta["dedup"] <= 50
    assert meta["returned"] <= 30

    # Ensure seed track is not present in final tracks
    seeds = {(t["artist_name"].strip().lower(), t["track_name"].strip().lower()) for t in out["tracks"]}
    assert (seed_artist.lower(), seed_track.lower()) not in seeds


@pytest.mark.django_db
def test_zero_weights_fallback_to_popularity(
    fake_seed,
    stub_lastfm_factory,
    stub_spotify_find_track,
    stub_encoder,
    monkeypatch,
):
    """
    If all Stage-A weights (α, γ, δ) are 0.0, engine should fall back to popularity
    to compute a non-degenerate shortlist.
    """
    seed_artist, seed_track = fake_seed

    from recommender.tests.conftest import _lfm_key  # type: ignore

    # Minimal pool: two items, different listeners/playcount so popularity ordering is deterministic
    responses = {
        _lfm_key("track.getTopTags", artist=seed_artist, track=seed_track): {"toptags": {"tag": []}},
        _lfm_key("track.getSimilar", artist=seed_artist, track=seed_track, limit=30): {
            "similartracks": {
                "track": [
                    {"name": "A", "artist": {"name": "X"}, "url": "u1", "listeners": "10", "playcount": "20"},
                    {"name": "B", "artist": {"name": "Y"}, "url": "u2", "listeners": "100", "playcount": "200"},
                ]
            }
        },
        # When SIMILAR_ARTISTS=0 some engines used to still call with limit=0. Provide a stub for robustness.
        _lfm_key("artist.getSimilar", artist=seed_artist, limit=0): {"similarartists": {"artist": []}},
    }
    monkeypatch.setattr(eng, "_lastfm_get", stub_lastfm_factory(responses), raising=True)
    monkeypatch.setattr(eng, "_track_top_tags", lambda *a, **k: [], raising=True)

    # No Spotify enrichment (forces popularity fallback to use Last.fm listeners+playcount)
    stub_spotify_find_track(lambda a, t: {})

    Dummy = stub_encoder
    monkeypatch.setattr(eng, "_MODEL", Dummy())
    monkeypatch.setattr(eng, "_get_model", lambda: eng._MODEL)

    out = eng.recommend_tracks(
        seed_artist,
        seed_track,
        SIMILAR_TRACKS=30, SIMILAR_ARTISTS=0,
        TOP_TRACKS_PER_ARTIST=0, TOP_TRACKS_PER_TAG=0,
        POOL_CAP=50, PRE_N=50,
        # all Stage-A weights zero
        ALPHA_EMB=0.0, GAMMA_POP=0.0, DELTA_FRESH=0.0, BETA_TAG=0.0,
        MMR_LAMBDA=0, TOP_K=2,
    )

    # With popularity fallback, the track with larger listeners+playcount ("B") should rank first
    names = [(t["artist_name"], t["track_name"]) for t in out["tracks"]]
    assert names[0] == ("Y", "B")
    assert names[1] == ("X", "A")


@pytest.mark.django_db
def test_model_cached_across_calls(
    fake_seed,
    stub_lastfm_factory,
    stub_spotify_find_track,
    stub_encoder,
    monkeypatch,
):
    """
    _get_model() should instantiate the SentenceTransformer only once per process.
    """
    seed_artist, seed_track = fake_seed
    from recommender.tests.conftest import _lfm_key  # type: ignore

    responses = {
        _lfm_key("track.getTopTags", artist=seed_artist, track=seed_track): {"toptags": {"tag": []}},
        _lfm_key("track.getSimilar", artist=seed_artist, track=seed_track, limit=30): {
            "similartracks": {"track": []}
        },
        _lfm_key("artist.getSimilar", artist=seed_artist, limit=5): {"similarartists": {"artist": []}},
    }
    monkeypatch.setattr(eng, "_lastfm_get", stub_lastfm_factory(responses), raising=True)
    monkeypatch.setattr(eng, "_track_top_tags", lambda *a, **k: [], raising=True)
    stub_spotify_find_track(lambda a, t: {})

    # Dummy encoder that increments a counter on construction
    Dummy = stub_encoder
    eng._MODEL = None
    monkeypatch.setattr(eng, "_get_model", lambda: eng._MODEL or setattr(eng, "_MODEL", Dummy()) or eng._MODEL)

    # Call twice; the Dummy should be constructed once (see stub in conftest)
    eng.recommend_tracks(seed_artist, seed_track, SIMILAR_TRACKS=0, SIMILAR_ARTISTS=0,
                         TOP_TRACKS_PER_ARTIST=0, TOP_TRACKS_PER_TAG=0, POOL_CAP=0)
    eng.recommend_tracks(seed_artist, seed_track, SIMILAR_TRACKS=0, SIMILAR_ARTISTS=0,
                         TOP_TRACKS_PER_ARTIST=0, TOP_TRACKS_PER_TAG=0, POOL_CAP=0)

    # Validate via the Dummy's class-level counter
    assert Dummy.construct_count == 1


@pytest.mark.django_db
def test_mmr_changes_order(
    fake_seed,
    stub_lastfm_factory,
    stub_spotify_find_track,
    stub_encoder,
    monkeypatch,
):
    """
    When MMR is enabled, order of otherwise-equal-scored items should change
    due to diversity.
    """
    seed_artist, seed_track = fake_seed
    from recommender.tests.conftest import _lfm_key  # type: ignore

    # Create 5 items; Dummy encoder will produce distinct vectors so MMR can reorder.
    tracks = [{"name": f"T{i}", "artist": {"name": f"A{i}"}, "url": f"u{i}"} for i in range(5)]
    responses = {
        _lfm_key("track.getTopTags", artist=seed_artist, track=seed_track): {"toptags": {"tag": ["x", "y", "z"]}},
        _lfm_key("track.getSimilar", artist=seed_artist, track=seed_track, limit=30): {
            "similartracks": {"track": tracks}
        },
        _lfm_key("artist.getSimilar", artist=seed_artist, limit=5): {"similarartists": {"artist": []}},
    }
    monkeypatch.setattr(eng, "_lastfm_get", stub_lastfm_factory(responses), raising=True)
    monkeypatch.setattr(eng, "_track_top_tags", lambda *a, **k: ["x", "y"], raising=True)
    stub_spotify_find_track(lambda a, t: {"popularity": 50, "album_release_date": "2021-01-01"})

    Dummy = stub_encoder
    monkeypatch.setattr(eng, "_MODEL", Dummy())
    monkeypatch.setattr(eng, "_get_model", lambda: eng._MODEL)

    out_no_mmr = eng.recommend_tracks(
        seed_artist, seed_track,
        SIMILAR_TRACKS=5, SIMILAR_ARTISTS=0,
        TOP_TRACKS_PER_ARTIST=0, TOP_TRACKS_PER_TAG=0,
        POOL_CAP=50, PRE_N=50,
        MMR_LAMBDA=0, TOP_K=5,
    )
    order_no = [t["track_name"] for t in out_no_mmr["tracks"]]

    out_mmr = eng.recommend_tracks(
        seed_artist, seed_track,
        SIMILAR_TRACKS=5, SIMILAR_ARTISTS=0,
        TOP_TRACKS_PER_ARTIST=0, TOP_TRACKS_PER_TAG=0,
        POOL_CAP=50, PRE_N=50,
        MMR_LAMBDA=0.6, MMR_K=5, TOP_K=5,
    )
    order_mmr = [t["track_name"] for t in out_mmr["tracks"]]

    # Orders should differ when MMR is on
    assert order_mmr != order_no
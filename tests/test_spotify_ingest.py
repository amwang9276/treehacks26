from spotify_ingest import _normalize_track, parse_playlist_id


def test_parse_playlist_id_from_url() -> None:
    url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc"
    assert parse_playlist_id(url) == "37i9dQZF1DXcBWIGoYBM5M"


def test_parse_playlist_id_raw_id() -> None:
    pid = "37i9dQZF1DX4JAvHpjipBk"
    assert parse_playlist_id(pid) == pid


def test_normalize_track_basic_fields() -> None:
    raw = {
        "track": {
            "id": "t1",
            "name": "Song",
            "artists": [{"name": "Artist A"}, {"name": "Artist B"}],
            "album": {"name": "Album"},
            "popularity": 55,
            "preview_url": "https://p.scdn.co/mp3-preview/test",
            "external_urls": {"spotify": "https://open.spotify.com/track/t1"},
            "duration_ms": 30000,
        }
    }
    out = _normalize_track(raw, playlist_id="p1")
    assert out is not None
    assert out.track_id == "t1"
    assert out.name == "Song"
    assert out.artists == ["Artist A", "Artist B"]
    assert out.album == "Album"
    assert out.preview_url is not None

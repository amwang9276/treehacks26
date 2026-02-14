from unittest.mock import patch

from es_index import SearchHit, search_by_text_embedding


class _FakeClient:
    def __init__(self, response):
        self._response = response

    def search(self, index, body):  # noqa: ANN001
        return self._response


def test_search_by_text_embedding_shapes_hits() -> None:
    response = {
        "hits": {
            "hits": [
                {
                    "_score": 0.91,
                    "_source": {
                        "track_id": "t1",
                        "name": "Track 1",
                        "artists": ["A"],
                        "album": "X",
                        "preview_url": "u1",
                        "spotify_url": "s1",
                    },
                }
            ]
        }
    }
    with patch("es_index._build_client", return_value=_FakeClient(response)):
        hits = search_by_text_embedding(
            "http://localhost:9200", "songs_mulan", [0.1, 0.2], top_k=1
        )
    assert len(hits) == 1
    assert isinstance(hits[0], SearchHit)
    assert hits[0].track_id == "t1"
    assert hits[0].score == 0.91

"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getSemanticIndexStatus, searchVibe, type SemanticIndexStatus, type VibeSearchItem } from "../../lib/api";

export default function VibePage() {
  const [text, setText] = useState("");
  const [topK, setTopK] = useState(10);
  const [status, setStatus] = useState<SemanticIndexStatus | null>(null);
  const [results, setResults] = useState<VibeSearchItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function loadStatus() {
      try {
        const s = await getSemanticIndexStatus();
        if (!mounted) return;
        setStatus(s);
      } catch (err) {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Failed to load index status");
      }
    }
    void loadStatus();
    return () => {
      mounted = false;
    };
  }, []);

  const ready = status?.status === "completed" && (status?.indexed || 0) > 0;

  return (
    <main className="container">
      <div className="card">
        <Link href="/playlists" className="btn">
          Back to playlists
        </Link>
      </div>
      <div className="card">
        <h1>Vibe Search</h1>
        <p className="muted">
          Describe the room mood and find semantically similar tracks from indexed Spotify previews.
        </p>
        <p className="muted">
          Index status: {status?.status ?? "unknown"} | Indexed tracks: {status?.indexed ?? 0}
        </p>
        {!ready && (
          <p className="muted">
            Index is not ready yet. Return to playlists and wait for indexing to complete.
          </p>
        )}
        <div style={{ display: "grid", gap: 12 }}>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. calm rainy evening study vibe"
            style={{ padding: 10, borderRadius: 8, border: "1px solid #d0d5dd" }}
          />
          <input
            type="number"
            value={topK}
            min={1}
            max={50}
            onChange={(e) => setTopK(Math.max(1, Math.min(50, Number(e.target.value) || 10)))}
            style={{ padding: 10, borderRadius: 8, border: "1px solid #d0d5dd", width: 120 }}
          />
          <button
            className="btn"
            disabled={!ready || !text.trim() || loading}
            onClick={async () => {
              setLoading(true);
              setError(null);
              try {
                const resp = await searchVibe(text.trim(), topK);
                setResults(resp.items);
              } catch (err) {
                setError(err instanceof Error ? err.message : "Search failed");
              } finally {
                setLoading(false);
              }
            }}
          >
            {loading ? "Searching..." : "Search vibe"}
          </button>
        </div>
      </div>

      {error && (
        <div className="card">
          <p className="muted">{error}</p>
        </div>
      )}

      {!loading && !error && results.length === 0 && ready && (
        <div className="card">
          <p className="muted">No results yet. Try a different vibe description.</p>
        </div>
      )}

      {!loading &&
        !error &&
        results.map((item) => (
          <div key={`${item.track_id}-${item.name}`} className="card">
            <h3>{item.name}</h3>
            <p className="muted">
              {(item.artists || []).join(", ")} | Album: {item.album || "Unknown"} | Score:{" "}
              {item.score.toFixed(4)}
            </p>
            <p className="muted">Playlists: {(item.playlist_names || []).join(", ") || "n/a"}</p>
            <p className="muted">Preview available: {item.preview_url ? "Yes" : "No"}</p>
            {item.spotify_url ? (
              <a className="btn" href={item.spotify_url} target="_blank" rel="noreferrer">
                Open in Spotify
              </a>
            ) : null}
          </div>
        ))}
    </main>
  );
}

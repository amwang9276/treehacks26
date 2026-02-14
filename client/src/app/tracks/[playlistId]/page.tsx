"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getPlaylistTracks, type Track } from "../../../lib/api";

export default function TracksPage({
  params,
}: {
  params: { playlistId: string };
}) {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const page = await getPlaylistTracks(params.playlistId);
        if (!mounted) return;
        setTracks(page.items);
      } catch (err) {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    void load();
    return () => {
      mounted = false;
    };
  }, [params.playlistId]);

  return (
    <main className="container">
      <div className="card">
        <Link href="/playlists" className="btn">
          Back to playlists
        </Link>
      </div>
      {loading && <p className="muted">Loading tracks...</p>}
      {error && (
        <div className="card">
          <p className="muted">{error}</p>
        </div>
      )}
      {!loading && !error && tracks.length === 0 && (
        <div className="card">
          <p className="muted">No tracks found in this playlist.</p>
        </div>
      )}
      {!loading &&
        !error &&
        tracks.map((track) => (
          <div key={`${track.track_id}-${track.name}`} className="card">
            <h3>{track.name}</h3>
            <p className="muted">
              {(track.artists || []).join(", ")} | Album: {track.album || "Unknown"}
            </p>
            <p className="muted">Preview available: {track.preview_url ? "Yes" : "No"}</p>
            {track.spotify_url ? (
              <a className="btn" href={track.spotify_url} target="_blank" rel="noreferrer">
                Open in Spotify
              </a>
            ) : null}
          </div>
        ))}
    </main>
  );
}

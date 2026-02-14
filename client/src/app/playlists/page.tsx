"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getMe, getPlaylists, logout, type MeResponse, type Playlist } from "../../lib/api";

export default function PlaylistsPage() {
  const [me, setMe] = useState<MeResponse | null>(null);
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const [profile, playlistPage] = await Promise.all([getMe(), getPlaylists()]);
        if (!mounted) return;
        setMe(profile);
        setPlaylists(playlistPage.items);
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
  }, []);

  if (loading) {
    return (
      <main className="container">
        <p className="muted">Loading playlists...</p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="container">
        <div className="card">
          <h2>Server offline or auth failed</h2>
          <p className="muted">{error}</p>
        </div>
      </main>
    );
  }

  const accessiblePlaylists = playlists.filter(
    (p) => Boolean(p.owner_id) && Boolean(me?.id) && p.owner_id === me?.id
  );

  return (
    <main className="container">
      <div className="card">
        <h1>Playlists</h1>
        <p className="muted">Connected as {me?.display_name || me?.id}</p>
        <button
          className="btn"
          onClick={async () => {
            await logout();
            window.location.href = "/";
          }}
        >
          Logout
        </button>
      </div>
      {accessiblePlaylists.length === 0 ? (
        <div className="card">
          <p className="muted">No track-accessible playlists found for this account.</p>
        </div>
      ) : (
        accessiblePlaylists.map((p) => (
          <div key={p.id} className="card">
            <h3>{p.name}</h3>
            <p className="muted">
              Owner: {p.owner} | Tracks: {p.tracks_total}
            </p>
            <Link href={`/tracks/${p.id}`} className="btn">
              View tracks
            </Link>
          </div>
        ))
      )}
    </main>
  );
}

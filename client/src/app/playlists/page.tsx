"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  getMe,
  getPlaylists,
  getSemanticIndexStatus,
  logout,
  startSemanticIndex,
  type MeResponse,
  type Playlist,
  type SemanticIndexStatus,
} from "../../lib/api";

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function phaseLabel(status: SemanticIndexStatus): string {
  switch (status.phase) {
    case "queued":
      return "Queued";
    case "initializing":
      return "Initializing";
    case "loading_model":
      return "Loading MuLan model";
    case "fetching_spotify":
      return "Fetching playlists and tracks from Spotify";
    case "embedding":
      return "Embedding previews and indexing";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    default:
      return "Idle";
  }
}

function progressPercent(status: SemanticIndexStatus): number {
  if (status.status === "completed") return 100;
  if (status.status === "failed") {
    if (status.total_to_index > 0) {
      return clampPercent((status.indexed / status.total_to_index) * 100);
    }
    return 0;
  }
  if (status.status === "idle") return 0;
  if (status.status === "queued") return 2;

  if (status.phase === "loading_model" || status.phase === "initializing") {
    return 10;
  }
  if (status.phase === "fetching_spotify") {
    if (status.playlist_total > 0) {
      const ratio = status.playlists_processed / status.playlist_total;
      return clampPercent(15 + ratio * 35);
    }
    return 20;
  }
  if (status.phase === "embedding") {
    if (status.total_to_index > 0) {
      const ratio = status.indexed / status.total_to_index;
      return clampPercent(55 + ratio * 45);
    }
    if (status.tracks_with_preview > 0) return 60;
    return 55;
  }
  return 5;
}

export default function PlaylistsPage() {
  const [me, setMe] = useState<MeResponse | null>(null);
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [indexStatus, setIndexStatus] = useState<SemanticIndexStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [indexBusy, setIndexBusy] = useState(false);
  const [playlistsLoaded, setPlaylistsLoaded] = useState(false);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const [profile, status] = await Promise.all([getMe(), getSemanticIndexStatus()]);
        if (!mounted) return;
        setMe(profile);
        setIndexStatus(status);
        if (!["queued", "running"].includes(status.status)) {
          const playlistPage = await getPlaylists();
          if (!mounted) return;
          setPlaylists(playlistPage.items);
          setPlaylistsLoaded(true);
        }
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

  useEffect(() => {
    if (!indexStatus || !["queued", "running"].includes(indexStatus.status)) {
      return;
    }
    let active = true;
    const timer = setInterval(async () => {
      try {
        const status = await getSemanticIndexStatus();
        if (!active) return;
        setIndexStatus(status);
      } catch {
        // keep previous state; next polling tick can retry
      }
    }, 3000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [indexStatus]);

  useEffect(() => {
    if (!indexStatus || playlistsLoaded) {
      return;
    }
    if (["queued", "running"].includes(indexStatus.status)) {
      return;
    }
    let active = true;
    async function loadPlaylistsAfterIndex() {
      try {
        const playlistPage = await getPlaylists();
        if (!active) return;
        setPlaylists(playlistPage.items);
        setPlaylistsLoaded(true);
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Failed to load playlists");
      }
    }
    void loadPlaylistsAfterIndex();
    return () => {
      active = false;
    };
  }, [indexStatus, playlistsLoaded]);

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
  const progress = indexStatus ? progressPercent(indexStatus) : 0;

  return (
    <main className="container">
      <div className="card">
        <h1>Playlists</h1>
        <p className="muted">Connected as {me?.display_name || me?.id}</p>
        <p>
          <Link href="/vibe" className="btn">
            Open vibe search
          </Link>
        </p>
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
      {indexStatus && (
        <div className="card">
          <h3>Semantic Index</h3>
          <p className="muted">Status: {indexStatus.status}</p>
          <p className="muted">
            Phase: {phaseLabel(indexStatus)} ({Math.round(progress)}%)
          </p>
          <div className="progress-track" role="progressbar" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100}>
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <p className="muted">
            Playlists: {indexStatus.playlists_scanned} | Tracks seen: {indexStatus.tracks_seen}
            {" | "}With preview: {indexStatus.tracks_with_preview} | Indexed: {indexStatus.indexed}
          </p>
          <p className="muted">
            Playlists processed: {indexStatus.playlists_processed}/{indexStatus.playlist_total}
            {" | "}Index target: {indexStatus.total_to_index || "pending"}
          </p>
          <p className="muted">
            Skipped (no preview): {indexStatus.skipped_no_preview} | Download failures:{" "}
            {indexStatus.download_failures} | Embed failures: {indexStatus.embed_failures}
          </p>
          {indexStatus.last_error && <p className="muted">Last error: {indexStatus.last_error}</p>}
          {["queued", "running"].includes(indexStatus.status) && (
            <p className="muted">
              Playlist fetch is temporarily deferred to reduce Spotify rate-limit errors while indexing runs.
            </p>
          )}
          {(indexStatus.status === "failed" || indexStatus.status === "idle") && (
            <button
              className="btn"
              disabled={indexBusy}
              onClick={async () => {
                setIndexBusy(true);
                try {
                  await startSemanticIndex();
                  const status = await getSemanticIndexStatus();
                  setIndexStatus(status);
                } catch (err) {
                  setError(err instanceof Error ? err.message : "Failed to restart indexing");
                } finally {
                  setIndexBusy(false);
                }
              }}
            >
              {indexBusy ? "Starting..." : "Start indexing"}
            </button>
          )}
        </div>
      )}
      {accessiblePlaylists.length === 0 ? (
        <div className="card">
          <p className="muted">
            {["queued", "running"].includes(indexStatus?.status || "")
              ? "Indexing in progress. Playlists will load after indexing completes."
              : "No track-accessible playlists found for this account."}
          </p>
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

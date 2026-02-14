"use client";

import { loginUrl } from "../lib/api";

export default function HomePage() {
  return (
    <main className="container">
      <div className="card">
        <h1>Spotify Connect</h1>
        <p className="muted">
          Connect your Spotify account so the backend can read your playlists and tracks.
        </p>
        <a className="btn" href={loginUrl()}>
          Connect Spotify
        </a>
      </div>
    </main>
  );
}

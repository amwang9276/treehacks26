"use client";

import Link from "next/link";
import { loginUrl } from "../lib/api";

export default function HomePage() {
  return (
    <main className="container">
      <div className="card">
        <h1>Choose Music Source</h1>
        <p className="muted">
          Select whether to use Suno-generated music or connect Spotify.
        </p>
      </div>

      <div className="source-grid">
        <div className="card">
          <h2>Suno</h2>
          <p className="muted">
            Generate new music dynamically from detected room mood and context.
          </p>
          <Link className="btn" href="/dashboard">
            Use Suno Generated Music
          </Link>
        </div>

        <div className="card">
          <h2>Spotify</h2>
          <p className="muted">
            Connect your Spotify account so the app can access your authorized music data.
          </p>
          <a className="btn" href={loginUrl()}>
            Connect Spotify
          </a>
        </div>
      </div>
    </main>
  );
}

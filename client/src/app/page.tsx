"use client";

import Link from "next/link";
import { loginUrl } from "../lib/api";

export default function HomePage() {
  return (
    <main className="container">
      <div className="card">
        <h1>Welcome to AuraLamp!</h1>
        <p className="muted">
          Select whether to use AI-generated music powered by Suno or connect Spotify to continue.
        </p>
      </div>

      <div className="source-grid">
        <div className="card">
          <h2>Feeling Adventurous</h2>
          <p className="muted">
            Generate new music dynamically from detected room mood and context.
          </p>
          <Link className="btn" href="/dashboard?mode=suno">
            Use AI Generated Music
          </Link>
        </div>

        <div className="card">
          <h2>The Classics</h2>
          <p className="muted">
            Connect your Spotify account so the app can access your authorized music data and you can enjoy your favorite tracks.
          </p>
          <a className="btn" href={loginUrl()}>
            Connect Spotify
          </a>
        </div>
      </div>
    </main>
  );
}

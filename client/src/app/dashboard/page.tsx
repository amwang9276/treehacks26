"use client";

import { useEffect } from "react";
import { useSearchParams } from "next/navigation";

function apiBase(): string {
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
}

export default function DashboardPage() {
  const params = useSearchParams();

  useEffect(() => {
    if (params.get("spotify_connected") === "1") {
      window.alert("Spotify successfully connected");
    }
  }, [params]);

  return (
    <main className="container">
      <div className="card">
        <h1>Dashboard</h1>
        <p className="muted">Live local camera feed (index 0)</p>
        <img
          src={`${apiBase()}/camera/stream?index=0&fps=15`}
          alt="Local camera stream"
          style={{
            width: "100%",
            maxWidth: 960,
            borderRadius: 12,
            border: "1px solid #e7eaf0",
            background: "#000",
          }}
        />
      </div>
    </main>
  );
}

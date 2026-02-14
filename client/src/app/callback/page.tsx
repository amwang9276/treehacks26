"use client";

import { useEffect } from "react";
import { useSearchParams } from "next/navigation";

function apiBase(): string {
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
}

export default function CallbackPage() {
  const params = useSearchParams();

  useEffect(() => {
    const qs = params.toString();
    const target = `${apiBase()}/auth/callback${qs ? `?${qs}` : ""}`;
    window.location.href = target;
  }, [params]);

  return (
    <main className="container">
      <div className="card">
        <h2>Completing Spotify login...</h2>
        <p className="muted">Redirecting to backend callback.</p>
      </div>
    </main>
  );
}

"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

function apiBase(): string {
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
}

const LOG_COLUMNS = ["SYSTEM", "EMOTION", "CONTEXT", "FUSION", "VOICE", "RETRIEVAL", "MUSIC"];

type RuntimeLogs = {
  running: boolean;
  [tag: string]: string[] | boolean;
};

export default function DashboardPage() {
  const params = useSearchParams();
  const [logs, setLogs] = useState<RuntimeLogs>({ running: false });
  const [error, setError] = useState<string | null>(null);
  const mode = params.get("mode");
  const generate = mode === "spotify" ? false : true;

  useEffect(() => {
    if (params.get("spotify_connected") === "1") {
      window.alert("Spotify successfully connected");
    }
  }, [params]);

  useEffect(() => {
    let cancelled = false;
    const start = async () => {
      try {
        await fetch(`${apiBase()}/dashboard/runtime/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ generate }),
          credentials: "include",
        });
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to start runtime");
        }
      }
    };
    start();
    return () => {
      cancelled = true;
      fetch(`${apiBase()}/dashboard/runtime/stop`, {
        method: "POST",
        credentials: "include",
      }).catch(() => {
        // noop
      });
    };
  }, [generate]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await fetch(`${apiBase()}/dashboard/runtime/logs`, {
          cache: "no-store",
          credentials: "include",
        });
        if (!res.ok) {
          throw new Error(`status ${res.status}`);
        }
        const payload = (await res.json()) as RuntimeLogs;
        if (!cancelled) {
          setLogs(payload);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load logs");
        }
      }
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const columns = useMemo(
    () =>
      LOG_COLUMNS.map((tag) => ({
        tag,
        entries: Array.isArray(logs[tag]) ? (logs[tag] as string[]) : [],
      })),
    [logs]
  );

  return (
    <main className="container">
      <div className="card">
        <h1>Dashboard</h1>
        <p className="muted">
          Runtime status: {logs.running ? "running" : "stopped"} | mode:{" "}
          {generate ? "suno" : "spotify retrieval"}
        </p>
        {error ? <p style={{ color: "#b42318" }}>Error: {error}</p> : null}
        <img
          src={`${apiBase()}/dashboard/runtime/stream?fps=15`}
          alt="Runtime camera stream"
          style={{
            width: "80%",
            maxWidth: 840,
            borderRadius: 12,
            border: "1px solid #e7eaf0",
            background: "#000",
            display: "block",
            margin: "0 auto",
          }}
        />
      </div>
      <section className="log-grid">
        {columns.map((column) => (
          <div key={column.tag} className="card">
            <h3>[{column.tag}]</h3>
            <div className="log-list">
              {column.entries.length === 0 ? (
                <p className="muted">No output yet.</p>
              ) : (
                column.entries.map((line, idx) => (
                  <p key={`${column.tag}-${idx}`} className="log-line">
                    {line}
                  </p>
                ))
              )}
            </div>
          </div>
        ))}
      </section>
    </main>
  );
}


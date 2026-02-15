# treehacks26

## Linux Setup

### Setup (pip + venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Setup from pyproject.toml

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Windows Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running The Repo

If using your local, laptop native webcam, run:

```bash
python main.py --source-type local --index 0
```

If you want to use a webcam attachment, run:

```bash
python main.py --source-type local --index 0
```

## MuLan + Spotify + Elasticsearch

### 1) Start Elasticsearch (local Docker)

```bash
docker compose -f docker-compose.elasticsearch.yml up -d
```

This project uses a dev-only Elasticsearch config with security disabled.

### 2) Set environment variables

```bash
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=songs_mulan
MULAN_MODEL_ID=OpenMuQ/MuQ-MuLan-large
```

### 3) Index a Spotify playlist (public)

```bash
python embed.py index-playlist --playlist-url <spotify_playlist_url>
```

### 4) Query by vibe text

```bash
python embed.py query-vibe --text "calm rainy evening study vibe" --top-k 10
```

## Client/Server Spotify Connect

This repo now includes:
1. `client/` (Next.js) for Spotify connect + playlist browsing
2. `server/` (FastAPI) for Spotify OAuth + playlist/track proxy APIs

### Server setup

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
```

Set environment variables:

```bash
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://127.0.0.1:3000/callback
CLIENT_ORIGIN=http://127.0.0.1:3000
SESSION_SECRET=replace_with_a_long_random_secret
```

Run backend:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Client setup

```bash
cd client
npm install
```

Create `client/.env.local`:

```bash
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

Run frontend:

```bash
npm run dev
```

Open `http://localhost:3000`.
In Spotify dashboard, add redirect URI exactly as:
`http://127.0.0.1:3000/callback`

Notes on playlist tracks:
1. Backend calls Spotify `GET /v1/playlists/{playlist_id}/items` (Get a Playlist's Items).
2. If Spotify returns `403 Forbidden`, the app now surfaces this as `SPOTIFY_FORBIDDEN` with a clear message.
3. Spotify may restrict this endpoint based on playlist access; in that case, use playlists your signed-in user can access.

## Semantic Vibe Search (Server + Client)

After Spotify login, the backend now automatically starts an async indexing job that:
1. Fetches all accessible playlists for the signed-in user
2. Reads playlist items and keeps tracks with non-null `preview_url`
3. Embeds preview audio with MuLan
4. Stores vectors and metadata in Elasticsearch index `songs_mulan_<spotify_user_id>`

Behavior note:
1. If that user index already exists with stored vectors, auto-start will reuse it and skip re-fetching Spotify on login.
2. `POST /semantic/index/start` still forces a fresh re-index.

New backend endpoints:
1. `POST /semantic/index/start`
2. `GET /semantic/index/status`
3. `GET /semantic/search?text=<vibe>&top_k=10`

New client route:
1. `http://127.0.0.1:3000/vibe`

MuLan runtime note:
1. `OpenMuQ/MuQ-MuLan-large` requires Python package `muq`.
2. If model loading fails with `No module named 'muq'`, install deps again:
   - root runtime: `pip install -r requirements.txt`
   - server runtime: `pip install -r server/requirements-server.txt`

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

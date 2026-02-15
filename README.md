# treehacks26

Camera + emotion + context/voice fusion + music orchestration.

- `Suno` mode: generate instrumental music from fusion prompt.
- `Spotify Retrieval` mode: select local songs via retrieval (MuLan + Elasticsearch fallback logic).
- Web dashboard streams camera output and categorized runtime logs (`[EMOTION]`, `[FUSION]`, `[MUSIC]`, etc.).

## 1) Root Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Environment Variables

Create `.env` in repo root with at least:

```env
OPENAI_API_KEY=...
SUNO_API_KEY=...
SUNO_BASE_URL=https://studio-api.prod.suno.com

SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://127.0.0.1:3000/callback
CLIENT_ORIGIN=http://127.0.0.1:3000
SESSION_SECRET=replace_with_a_long_random_secret
SPOTIFY_SCOPES=playlist-read-private playlist-read-collaborative

ELASTICSEARCH_URL=...
ELASTICSEARCH_API_KEY=...   # or ELASTICSEARCH_USERNAME/ELASTICSEARCH_PASSWORD
ELASTICSEARCH_VERIFY_CERTS=true
ELASTICSEARCH_INDEX=lyrics
MULAN_MODEL_ID=OpenMuQ/MuQ-MuLan-large
```

## 3) Run Web App (Recommended)

### Backend

```powershell
cd server
python -m pip install -r requirements-server.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```powershell
cd client
npm install
```

Create `client/.env.local`:

```env
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

Then:

```powershell
npm run dev
```

Open `http://127.0.0.1:3000`.

## 4) Dashboard Behavior

- Choose source on `/`:
  - `Suno`
  - `Connect Spotify`
- On `/dashboard`, runtime starts once and keeps camera/emotion/context/voice running.
- Toggle `Suno` / `Spotify Retrieval` on dashboard switches only the music branch (no full runtime restart).
- Dashboard startup also syncs Elasticsearch index `lyrics` from local `lyrics/*.txt` (upsert + stale delete).

## 5) Run Main CLI

```powershell
python main.py --source-type local --index 0
```

Useful flags:

```powershell
python main.py --generate true  --source-type local --index 0 --stable-seconds 1 --suno-poll-interval 2.5
python main.py --generate false --source-type local --index 0 --stable-seconds 1
```

## 6) Lyrics Index Sync (Manual)

To manually sync lyrics files to Elasticsearch:

```powershell
python es_index.py --index-name lyrics --lyrics-dir lyrics
```

This now:
- uploads/updates docs for current lyric files
- deletes stale docs no longer present in local folder

Disable delete-sync:

```powershell
python es_index.py --index-name lyrics --lyrics-dir lyrics --no-sync-delete
```

## 7) Retrieval / MuLan Notes

- Default model: `OpenMuQ/MuQ-MuLan-large`
- Requires `muq` package and compatible `torch` runtime.
- If MuLan fails during retrieval initialization, runtime now falls back to Elastic-only local-song selection instead of hard failing.

## 8) Spotify OAuth Notes

- Spotify app redirect URI must exactly match:
  - `http://127.0.0.1:3000/callback`
- In dev mode, Spotify may restrict accounts unless app/user settings are configured in Spotify Developer Dashboard.


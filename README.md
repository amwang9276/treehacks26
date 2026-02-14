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

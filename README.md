# treehacks26

## Setup (pip + venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then run:

```bash
python camera.py
```

## Setup from pyproject.toml

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python camera.py
```
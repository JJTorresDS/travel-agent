# travel-agent

## Setup instructions

Open your terminal

```bash
uv init
```

This creates a new project with a basic structure:

```
.
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
└── main.py (or hello.py depending on version)
```

Next, add dependencies:

```bash
uv add requests
```

This updates `pyproject.toml` and creates/updates the `uv.lock` file.

Run your script:

```bash
uv run main.py
```


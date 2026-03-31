## Setting up with uv (Windows)

**Install uv:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installing, restart your terminal so `uv` is on your PATH.

**Clone and sync:**

```powershell
git clone
cd <project>
uv sync
```

This creates a `.venv` and installs all locked dependencies — no need to run `pip install` manually.

**Activate the virtual environment (optional):**

uv commands auto-use the `.venv`, so activation isn't strictly needed. But if you want it for running scripts directly:

```powershell
.venv\Scripts\activate
```

**Run scripts without activating:**

```powershell
uv run python main.py
uv run pytest
```

**Add or remove a dependency:**

```powershell
uv add requests
uv remove requests
```

> **Important:** Do not use `pip install` directly — it bypasses `pyproject.toml` and `uv.lock`, causing environment drift. Always use `uv add` / `uv sync`.

---

Want me to save this as a file, or should I also add Linux/macOS instructions alongside it?

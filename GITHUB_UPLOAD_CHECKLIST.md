# GitHub Upload Checklist

## Include in GitHub

- `app.py`
- `rag_service.py`
- `llm_service.py`
- `requirements.txt`
- `Dockerfile`
- `.dockerignore`
- `.gitignore`
- `.env.example`
- `README.md`
- `static/`
- `scripts/`
- `rag_sources/`

## Do not include in GitHub

These files are local/runtime artifacts or too large for a normal GitHub repo.

- `.env`
- `.venv/`
- `data/`
- `logs/`
- `vector_db/`
- `qwen3_e1/`
- `models/`
- `*.sqlite3`
- `*.npy`
- `*.bin`
- `*.faiss`
- `*.pkl`
- `*.pickle`
- `*.tar`
- `*.zip`

## Large RAG files

Keep the Qwen3 E1 artifact outside GitHub and share it separately.

Expected runtime layout:

```text
qwen3_e1/
  embeddings.npy
  chunks.jsonl
  manifest.json
```

For real Qwen3 vector search, also provide the model separately:

```text
models/
  Qwen3-Embedding-4B/
```

Without the model, the service falls back to SQLite FTS over `chunks.jsonl`.

## First commit

```bash
git init
git add .
git status
git commit -m "Initial SMU chatbot MVP"
```

Before committing, confirm that `git status` does not show `.env`, `data/`,
`qwen3_e1/`, `vector_db/`, or `models/`.

## Push to GitHub

```bash
git branch -M main
git remote add origin https://github.com/<your-id>/<repo-name>.git
git push -u origin main
```

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install requirements.txt first."
        ) from exc

    model_id = os.environ.get("SMU_E1_MODEL_ID", "Qwen/Qwen3-Embedding-4B")
    target = Path(os.environ.get("SMU_E1_MODEL_DIR", "models/Qwen3-Embedding-4B")).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {model_id} to {target.resolve()}")
    print(f'Set SMU_E1_MODEL_DIR="{target.resolve()}" before starting app.py.')


if __name__ == "__main__":
    main()

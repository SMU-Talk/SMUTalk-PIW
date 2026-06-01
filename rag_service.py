from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOCAL_INDEX_PATH = DATA_DIR / "rag_index.sqlite3"
E1_FTS_INDEX_PATH = DATA_DIR / "qwen3_e1_fts.sqlite3"

DEFAULT_COLLECTION = "smu_notices_crawled_20260509_110005_qwen3_embedding_0_6b"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TASK_PROMPT = (
    "Retrieve relevant Sangmyung University notice passages for answering "
    "a Korean chatbot question."
)


@dataclass
class SearchHit:
    document: str
    metadata: dict[str, Any]
    score: float | None = None


BUILTIN_DOCUMENTS = [
    {
        "title": "상명대학교 공지 확인 안내",
        "date": "상시",
        "source_name": "SMU Talk 기본 문서",
        "url": "https://www.smu.ac.kr/",
        "text": (
            "상명대학교의 학사, 장학, 등록, 비교과, 취업, 도서관, 포털 관련 정보는 "
            "학교 공식 홈페이지와 포털, 단과대 및 학과 공지에서 확인해야 합니다. "
            "일정과 세부 기준은 매 학기 변경될 수 있으므로 공지의 작성일, 대상, 신청 기간, "
            "담당 부서를 함께 확인하는 것이 중요합니다."
        ),
    },
    {
        "title": "장학금 및 등록 관련 확인 방법",
        "date": "상시",
        "source_name": "SMU Talk 기본 문서",
        "url": "https://www.smu.ac.kr/",
        "text": (
            "장학금과 등록금 관련 질문은 국가장학, 교내장학, 교외장학, 등록 기간, "
            "납부 방법, 제출 서류를 구분해서 확인해야 합니다. 신청 기간을 놓치지 않도록 "
            "상명대학교 홈페이지 공지사항과 포털의 장학 또는 등록 메뉴를 함께 확인하세요."
        ),
    },
    {
        "title": "학사 일정과 수강신청 확인 방법",
        "date": "상시",
        "source_name": "SMU Talk 기본 문서",
        "url": "https://www.smu.ac.kr/",
        "text": (
            "수강신청, 휴학, 복학, 졸업, 성적, 계절학기 등 학사 업무는 학사 일정과 "
            "소속 캠퍼스 및 학과 기준을 확인해야 합니다. 공지의 대상 학년, 전공 제한, "
            "신청 시스템, 마감 시간을 함께 확인하는 것이 좋습니다."
        ),
    },
    {
        "title": "포털 및 계정 문제 확인 방법",
        "date": "상시",
        "source_name": "SMU Talk 기본 문서",
        "url": "https://portal.smu.ac.kr/",
        "text": (
            "상명대학교 포털, 샘물, e-Campus 로그인 문제는 계정 상태, 비밀번호, "
            "브라우저 캐시, 인증 방식, 학교 시스템 점검 여부를 확인해야 합니다. "
            "반복해서 실패하면 학교 IT 지원 또는 담당 행정 부서에 문의하세요."
        ),
    },
]


def _split_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(item.strip()).expanduser() for item in re.split(r"[;,]", value) if item.strip()]


def _find_chroma_dir() -> Path | None:
    configured = os.environ.get("SMU_CHROMA_PATH")
    if configured:
        path = Path(configured).expanduser()
        if (path / "chroma.sqlite3").exists():
            return path

    candidate_roots = [
        BASE_DIR / "vector_db",
        BASE_DIR / "data" / "vector_db",
        Path.home() / "Downloads",
        Path.home() / "Documents",
        Path.home() / "OneDrive",
    ]
    patterns = [
        "**/chroma.sqlite3",
        "**/chroma_qwen3_embedding_0_6b_smu_notices_crawled_*/chroma.sqlite3",
        "**/smu*notice*/**/chroma.sqlite3",
        "**/vector_db/**/chroma.sqlite3",
    ]

    matches: list[Path] = []
    for root in candidate_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            try:
                matches.extend(root.glob(pattern))
            except OSError:
                continue

    matches = sorted(set(matches), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].parent if matches else None


def _load_build_info(chroma_dir: Path) -> dict[str, Any]:
    build_info_path = chroma_dir / "build_info.json"
    info: dict[str, Any] = {}
    if build_info_path.exists():
        try:
            info.update(json.loads(build_info_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass

    sqlite_path = chroma_dir / "chroma.sqlite3"
    if sqlite_path.exists():
        try:
            with sqlite3.connect(sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT name, dimension FROM collections LIMIT 1").fetchone()
                if row:
                    info.setdefault("collection_name", row["name"])
                    info.setdefault("embedding_dimension", row["dimension"])
                count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                info.setdefault("chunk_count", count)
        except sqlite3.Error:
            pass
    return info


def _safe_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", text)
    stopwords = {
        "어디서",
        "어떻게",
        "알려줘",
        "확인",
        "되나요",
        "관련",
        "정보",
        "상명대학교",
        "상명대",
    }
    return [token for token in tokens if token not in stopwords][:10]


def _fts_query(text: str) -> str:
    tokens = _safe_tokens(text)
    if not tokens:
        tokens = [text.strip()]
    return " OR ".join(f'"{token}"' for token in tokens if token.strip())


def _metadata_from_document(document: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for raw_line in document.splitlines():
        line = raw_line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                mapped = {
                    "title": "title",
                    "url": "url",
                    "date": "date",
                    "source": "source_name",
                    "writer": "writer",
                    "article_no": "article_no",
                }.get(key)
                if mapped and value and mapped not in metadata:
                    metadata[mapped] = value
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("title", "notice_title", "url", "notice_url", "date", "notice_date", "source_name", "source_scope", "article_no"):
            if key in item and item[key]:
                normalized = {
                    "notice_title": "title",
                    "notice_url": "url",
                    "notice_date": "date",
                }.get(key, key)
                if normalized not in metadata:
                    metadata[normalized] = item[key]
    return metadata


def _clean_excerpt(document: str, max_chars: int = 520) -> str:
    text = re.sub(r"\s+", " ", document).strip()
    text = re.sub(r'"path":\s*"[^"]+",?\s*', "", text)
    text = re.sub(r'"error":\s*"[^"]+",?\s*', "", text)
    return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")


class LocalRagService:
    def __init__(self) -> None:
        self.e1_artifact_dir = self._find_e1_artifact_dir()
        self.e1_manifest = self._load_e1_manifest()
        self.e1_index_path = Path(os.environ.get("SMU_E1_INDEX_PATH", str(E1_FTS_INDEX_PATH))).expanduser()
        self.chroma_dir = _find_chroma_dir()
        self.build_info = _load_build_info(self.chroma_dir) if self.chroma_dir else {}
        self.collection_name = os.environ.get(
            "SMU_CHROMA_COLLECTION",
            self.build_info.get("collection_name", DEFAULT_COLLECTION),
        )
        self.model_name = os.environ.get(
            "SMU_EMBED_MODEL",
            self.build_info.get("model_name", DEFAULT_MODEL),
        )
        self.local_index_path = Path(os.environ.get("SMU_RAG_INDEX_PATH", str(LOCAL_INDEX_PATH))).expanduser()
        self.source_dirs = self._source_dirs()
        self._collection = None
        self._embedder = None
        self._vector_error: str | None = None
        self._local_error: str | None = None
        self._e1_vector_error: str | None = None
        self._e1_fts_error: str | None = None
        self._e1_embeddings = None
        self._e1_chunks: dict[int, dict[str, Any]] = {}
        self._ensure_local_index()
        self._ensure_e1_fts_index()

    def _find_e1_artifact_dir(self) -> Path | None:
        configured = os.environ.get("SMU_E1_ARTIFACT_DIR")
        candidates = _split_paths(configured)
        candidates.extend(
            [
                BASE_DIR / "qwen3_e1",
                BASE_DIR / "smu_notice_qwen3_e1_release",
                BASE_DIR / "data" / "qwen3_e1",
            ]
        )
        for path in candidates:
            if (path / "embeddings.npy").exists() and (path / "chunks.jsonl").exists():
                return path
        return None

    def _load_e1_manifest(self) -> dict[str, Any]:
        if not self.e1_artifact_dir:
            return {}
        manifest_path = self.e1_artifact_dir / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _source_dirs(self) -> list[Path]:
        configured = _split_paths(os.environ.get("SMU_RAG_SOURCE_DIRS"))
        if configured:
            return configured
        return [
            BASE_DIR / "rag_sources",
            BASE_DIR / "notices",
            DATA_DIR / "rag_sources",
        ]

    def status(self) -> dict[str, Any]:
        sqlite_path = self.chroma_dir / "chroma.sqlite3" if self.chroma_dir else None
        local_count = self._local_document_count()
        e1_count = self._e1_document_count()
        e1_embedding_shape = self._e1_embedding_shape()
        return {
            "enabled": self.e1_artifact_dir is not None or self.chroma_dir is not None or local_count > 0,
            "mode": self._mode(local_count, e1_count),
            "e1_artifact_path": str(self.e1_artifact_dir) if self.e1_artifact_dir else None,
            "e1_model_name": self.e1_manifest.get("model_id") or os.environ.get("SMU_E1_MODEL_ID", "Qwen/Qwen3-Embedding-4B"),
            "e1_model_ref": self._e1_model_ref(),
            "e1_embedding_dimension": self.e1_manifest.get("embedding_dim"),
            "e1_embedding_shape": e1_embedding_shape,
            "e1_chunk_count": self.e1_manifest.get("chunk_count"),
            "e1_fts_index_path": str(self.e1_index_path),
            "e1_fts_document_count": e1_count,
            "e1_vector_ready": self._e1_vector_ready(),
            "e1_vector_error": self._e1_vector_error,
            "e1_fts_error": self._e1_fts_error,
            "chroma_path": str(self.chroma_dir) if self.chroma_dir else None,
            "sqlite_exists": bool(sqlite_path and sqlite_path.exists()),
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "source_document_count": self.build_info.get("source_document_count"),
            "chunk_count": self.build_info.get("chunk_count"),
            "embedding_dimension": self.build_info.get("embedding_dimension"),
            "vector_ready": self._vector_ready(),
            "vector_error": self._vector_error,
            "local_index_path": str(self.local_index_path),
            "local_document_count": local_count,
            "local_source_dirs": [str(path) for path in self.source_dirs],
            "local_error": self._local_error,
        }

    def _mode(self, local_count: int, e1_count: int) -> str:
        if self.e1_artifact_dir:
            if self._e1_vector_ready():
                return "qwen3_e1_vector"
            if e1_count > 0:
                return "qwen3_e1_fts"
        if self.chroma_dir:
            return "chroma_vector_or_sqlite"
        if local_count > 0:
            return "local_sqlite_fts"
        return "disabled"

    def reload(self) -> dict[str, Any]:
        self._ensure_local_index(force=True)
        self._ensure_e1_fts_index(force=True)
        return self.status()

    def answer(self, question: str, top_k: int = 5) -> str | None:
        hits: list[SearchHit] = []
        mode = "로컬 문서 검색"

        if self._is_navigation_query(question):
            hits = self._local_fts_search(question, top_k=top_k)
            if hits:
                return self._format_answer(hits, "학교 홈페이지 메뉴 안내")

        if self.e1_artifact_dir:
            hits = self._e1_vector_search(question, top_k=top_k)
            mode = "Qwen3-Embedding-4B 벡터 검색"
            if not hits:
                hits = self._e1_fts_search(question, top_k=top_k)
                mode = "Qwen3 E1 문서 검색"
            if hits:
                return self._format_answer(hits, mode)

        if self.chroma_dir:
            hits = self._vector_search(question, top_k=top_k)
            mode = "벡터 검색"
            if not hits:
                hits = self._chroma_sqlite_fts_search(question, top_k=top_k)
                mode = "Chroma 문서 검색"

        if not hits:
            hits = self._local_fts_search(question, top_k=top_k)
            mode = "로컬 문서 검색"

        if not hits:
            return None

        return self._format_answer(hits, mode)

    @staticmethod
    def _is_navigation_query(question: str) -> bool:
        keywords = [
            "상명소개",
            "입학안내",
            "대학ㆍ대학원",
            "대학·대학원",
            "연구ㆍ산학",
            "연구·산학",
            "학사안내",
            "대학생활",
            "전공제도",
            "수업 및 수강신청",
            "학적변동",
            "장학금지급규정",
            "교내 장학금",
            "교외 장학금",
            "버스안내",
            "학생증발급",
            "전자출결시스템",
        ]
        return any(keyword in question for keyword in keywords)

    def _e1_paths(self) -> tuple[Path, Path] | None:
        if not self.e1_artifact_dir:
            return None
        embeddings_path = self.e1_artifact_dir / "embeddings.npy"
        chunks_path = self.e1_artifact_dir / "chunks.jsonl"
        if not embeddings_path.exists() or not chunks_path.exists():
            return None
        return embeddings_path, chunks_path

    def _e1_vector_enabled(self) -> bool:
        return os.environ.get("SMU_E1_ENABLE_VECTOR", "true").strip().lower() in {"1", "true", "yes", "on"}

    def _e1_vector_ready(self) -> bool:
        paths = self._e1_paths()
        if not paths:
            return False
        if not self._e1_vector_enabled():
            self._e1_vector_error = "disabled by SMU_E1_ENABLE_VECTOR"
            return False
        model_dir = os.environ.get("SMU_E1_MODEL_DIR", "").strip()
        allow_download = os.environ.get("SMU_E1_ALLOW_MODEL_DOWNLOAD", "false").strip().lower() in {"1", "true", "yes", "on"}
        if model_dir and not Path(model_dir).expanduser().exists():
            self._e1_vector_error = f"SMU_E1_MODEL_DIR does not exist: {model_dir}"
            return False
        if not self._existing_e1_model_dir() and not allow_download:
            self._e1_vector_error = (
                "Qwen3-Embedding-4B model files are not available locally. "
                "Set SMU_E1_MODEL_DIR to a downloaded model directory or set "
                "SMU_E1_ALLOW_MODEL_DOWNLOAD=true."
            )
            return False
        try:
            import numpy  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except Exception as exc:
            self._e1_vector_error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def _load_e1_embeddings(self):
        if self._e1_embeddings is not None:
            return self._e1_embeddings
        paths = self._e1_paths()
        if not paths:
            return None
        try:
            import numpy as np

            self._e1_embeddings = np.load(paths[0], mmap_mode="r")
            return self._e1_embeddings
        except Exception as exc:
            self._e1_vector_error = f"{type(exc).__name__}: {exc}"
            return None

    def _e1_embedding_shape(self) -> list[int] | None:
        embeddings = self._load_e1_embeddings()
        if embeddings is None:
            return None
        return [int(value) for value in embeddings.shape]

    def _e1_model_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        configured = os.environ.get("SMU_E1_MODEL_DIR", "").strip()
        if configured:
            candidates.append(Path(configured).expanduser())

        manifest_model_dir = str(self.e1_manifest.get("model_dir") or "").strip()
        if manifest_model_dir:
            manifest_path = Path(manifest_model_dir)
            if manifest_path.is_absolute():
                candidates.append(manifest_path)
            candidates.extend(
                [
                    BASE_DIR / manifest_path,
                    self.e1_artifact_dir / manifest_path if self.e1_artifact_dir else manifest_path,
                    self.e1_artifact_dir.parent / manifest_path if self.e1_artifact_dir else manifest_path,
                ]
            )

        candidates.extend(
            [
                BASE_DIR / "models" / "Qwen3-Embedding-4B",
                DATA_DIR / "models" / "Qwen3-Embedding-4B",
                Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-Embedding-4B",
            ]
        )

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                unique.append(candidate)
                seen.add(key)
        return unique

    def _existing_e1_model_dir(self) -> Path | None:
        for candidate in self._e1_model_candidates():
            if not candidate.exists():
                continue
            if (candidate / "config.json").exists():
                return candidate
            snapshots_dir = candidate / "snapshots"
            if snapshots_dir.exists():
                snapshots = sorted(
                    [path for path in snapshots_dir.iterdir() if (path / "config.json").exists()],
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if snapshots:
                    return snapshots[0]
        return None

    def _e1_model_ref(self) -> str:
        model_dir = self._existing_e1_model_dir()
        if model_dir:
            return str(model_dir)
        return os.environ.get("SMU_E1_MODEL_ID", self.e1_manifest.get("model_id", "Qwen/Qwen3-Embedding-4B"))

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_e1_model_cached(model_ref: str, device: str, max_seq_length: int):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = device
        model_kwargs: dict[str, Any] = {}
        if selected_device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        model = SentenceTransformer(
            model_ref,
            device=selected_device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
        )
        model.max_seq_length = max_seq_length
        return model

    def _load_e1_model(self):
        model_ref = self._e1_model_ref()
        device = os.environ.get("SMU_E1_DEVICE", "auto")
        max_seq_length = int(os.environ.get("SMU_E1_MAX_SEQ_LENGTH", "2048"))
        return self._load_e1_model_cached(model_ref, device, max_seq_length)

    def _e1_chunk_by_index(self, index: int) -> dict[str, Any]:
        if index in self._e1_chunks:
            return self._e1_chunks[index]
        paths = self._e1_paths()
        if not paths:
            return {}
        with paths[1].open("r", encoding="utf-8") as f:
            for row_index, line in enumerate(f):
                if row_index == index:
                    item = json.loads(line)
                    self._e1_chunks[index] = item
                    return item
        return {}

    def _e1_vector_search(self, question: str, top_k: int) -> list[SearchHit]:
        if not self._e1_vector_ready():
            return []
        embeddings = self._load_e1_embeddings()
        if embeddings is None:
            return []
        try:
            import numpy as np

            model = self._load_e1_model()
            query_vec = model.encode(
                [question],
                batch_size=1,
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0].astype("float32")

            best: list[tuple[float, int]] = []
            batch_size = int(os.environ.get("SMU_E1_SEARCH_BATCH_SIZE", "4096"))
            for start in range(0, embeddings.shape[0], batch_size):
                end = min(start + batch_size, embeddings.shape[0])
                scores = np.asarray(embeddings[start:end], dtype="float32") @ query_vec
                if scores.size == 0:
                    continue
                take = min(top_k, scores.size)
                ids = np.argpartition(-scores, take - 1)[:take]
                best.extend((float(scores[i]), start + int(i)) for i in ids)
                best = sorted(best, key=lambda item: item[0], reverse=True)[:top_k]

            hits: list[SearchHit] = []
            for score, idx in best:
                chunk = self._e1_chunk_by_index(idx)
                if chunk:
                    hits.append(self._e1_hit_from_chunk(chunk, score))
            return hits
        except Exception as exc:
            self._e1_vector_error = f"{type(exc).__name__}: {exc}"
            return []

    def _e1_hit_from_chunk(self, chunk: dict[str, Any], score: float | None = None) -> SearchHit:
        metadata = {
            "title": chunk.get("notice_title") or chunk.get("title") or f"청크 {chunk.get('chunk_id', '')}",
            "url": chunk.get("notice_url") or chunk.get("source_url") or "",
            "date": chunk.get("notice_date") or "",
            "source_name": chunk.get("source_name") or "",
            "source_scope": chunk.get("source_scope") or "",
            "article_no": chunk.get("article_no") or "",
        }
        return SearchHit(document=str(chunk.get("text") or ""), metadata=metadata, score=score)

    def _ensure_e1_fts_index(self, force: bool = False) -> None:
        paths = self._e1_paths()
        if not paths:
            return
        chunks_path = paths[1]
        self.e1_index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(self.e1_index_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS e1_chunks (
                        id INTEGER PRIMARY KEY,
                        chunk_json TEXT NOT NULL,
                        title TEXT,
                        url TEXT,
                        date TEXT,
                        source_name TEXT,
                        content TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS e1_chunks_fts
                    USING fts5(title, content, source_name, content='e1_chunks', content_rowid='id')
                    """
                )
                existing = int(conn.execute("SELECT COUNT(*) FROM e1_chunks").fetchone()[0])
                expected = int(self.e1_manifest.get("chunk_count") or 0)
                if not force and existing > 0 and (expected == 0 or existing == expected):
                    return
                conn.execute("DELETE FROM e1_chunks_fts")
                conn.execute("DELETE FROM e1_chunks")
                batch: list[tuple[Any, ...]] = []
                fts_batch: list[tuple[Any, ...]] = []
                with chunks_path.open("r", encoding="utf-8") as f:
                    for row_index, line in enumerate(f):
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        chunk_id = int(chunk.get("chunk_id", row_index))
                        title = str(chunk.get("notice_title") or chunk.get("title") or "")
                        url = str(chunk.get("notice_url") or chunk.get("source_url") or "")
                        date = str(chunk.get("notice_date") or "")
                        source_name = str(chunk.get("source_name") or "")
                        content = str(chunk.get("text") or "")
                        batch.append((chunk_id, json.dumps(chunk, ensure_ascii=False), title, url, date, source_name, content))
                        fts_batch.append((chunk_id, title, content, source_name))
                        if len(batch) >= 1000:
                            conn.executemany(
                                "INSERT INTO e1_chunks (id, chunk_json, title, url, date, source_name, content) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                batch,
                            )
                            conn.executemany(
                                "INSERT INTO e1_chunks_fts (rowid, title, content, source_name) VALUES (?, ?, ?, ?)",
                                fts_batch,
                            )
                            batch.clear()
                            fts_batch.clear()
                    if batch:
                        conn.executemany(
                            "INSERT INTO e1_chunks (id, chunk_json, title, url, date, source_name, content) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            batch,
                        )
                        conn.executemany(
                            "INSERT INTO e1_chunks_fts (rowid, title, content, source_name) VALUES (?, ?, ?, ?)",
                            fts_batch,
                        )
        except Exception as exc:
            self._e1_fts_error = f"{type(exc).__name__}: {exc}"

    def _e1_document_count(self) -> int:
        if not self.e1_index_path.exists():
            return 0
        try:
            with sqlite3.connect(self.e1_index_path) as conn:
                return int(conn.execute("SELECT COUNT(*) FROM e1_chunks").fetchone()[0])
        except sqlite3.Error:
            return 0

    def _e1_fts_search(self, question: str, top_k: int) -> list[SearchHit]:
        if not self.e1_index_path.exists():
            return []
        query = _fts_query(question)
        try:
            with sqlite3.connect(self.e1_index_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT chunk_json, bm25(e1_chunks_fts) AS score
                    FROM e1_chunks_fts
                    JOIN e1_chunks c ON c.id = e1_chunks_fts.rowid
                    WHERE e1_chunks_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (query, top_k),
                ).fetchall()
        except sqlite3.Error as exc:
            self._e1_fts_error = f"{type(exc).__name__}: {exc}"
            return []

        hits: list[SearchHit] = []
        for row in rows:
            chunk = json.loads(row["chunk_json"])
            hits.append(self._e1_hit_from_chunk(chunk, float(row["score"]) if row["score"] is not None else None))
        return hits

    def _vector_ready(self) -> bool:
        try:
            import chromadb  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on local packages
            self._vector_error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def _load_vector_components(self) -> bool:
        if self._collection is not None and self._embedder is not None:
            return True
        if not self._vector_ready() or not self.chroma_dir:
            return False

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self._collection = client.get_collection(self.collection_name)
            self._embedder = SentenceTransformer(self.model_name)
            return True
        except Exception as exc:  # pragma: no cover - depends on local packages/model
            self._vector_error = f"{type(exc).__name__}: {exc}"
            self._collection = None
            self._embedder = None
            return False

    def _embed_question(self, question: str) -> list[float]:
        query = f"Instruct: {TASK_PROMPT}\nQuery: {question}"
        embedding = self._embedder.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    def _vector_search(self, question: str, top_k: int) -> list[SearchHit]:
        if not self._load_vector_components():
            return []
        try:
            results = self._collection.query(
                query_embeddings=[self._embed_question(question)],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # pragma: no cover - depends on local packages/model
            self._vector_error = f"{type(exc).__name__}: {exc}"
            return []

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        hits: list[SearchHit] = []
        for index, document in enumerate(documents):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            metadata.update({k: v for k, v in _metadata_from_document(document).items() if k not in metadata})
            score = distances[index] if index < len(distances) else None
            hits.append(SearchHit(document=document, metadata=metadata, score=score))
        return hits

    def _chroma_sqlite_fts_search(self, question: str, top_k: int) -> list[SearchHit]:
        if not self.chroma_dir:
            return []
        sqlite_path = self.chroma_dir / "chroma.sqlite3"
        if not sqlite_path.exists():
            return []

        query = _fts_query(question)
        try:
            with sqlite3.connect(sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT rowid, string_value, bm25(embedding_fulltext_search) AS score
                    FROM embedding_fulltext_search
                    WHERE string_value MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (query, top_k),
                ).fetchall()
        except sqlite3.Error:
            rows = []

        return [
            SearchHit(
                document=str(row["string_value"]),
                metadata=_metadata_from_document(str(row["string_value"])),
                score=float(row["score"]) if row["score"] is not None else None,
            )
            for row in rows
        ]

    def _ensure_local_index(self, force: bool = False) -> None:
        self.local_index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            documents = self._load_source_documents()
            if not documents:
                documents = BUILTIN_DOCUMENTS

            with sqlite3.connect(self.local_index_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_path TEXT,
                        title TEXT NOT NULL,
                        url TEXT,
                        date TEXT,
                        source_name TEXT,
                        metadata TEXT,
                        content TEXT NOT NULL
                    )
                    """
                )
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(rag_documents)").fetchall()
                }
                if "metadata" not in columns:
                    conn.execute("ALTER TABLE rag_documents ADD COLUMN metadata TEXT")
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS rag_documents_fts
                    USING fts5(title, content, metadata, content='rag_documents', content_rowid='id')
                    """
                )
                count = conn.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]
                should_rebuild = force or count == 0 or (documents and count != len(documents))
                if should_rebuild:
                    conn.execute("DELETE FROM rag_documents_fts")
                    conn.execute("DELETE FROM rag_documents")
                    for doc in documents:
                        metadata = json.dumps(
                            {
                                "title": doc.get("title") or "문서",
                                "url": doc.get("url", ""),
                                "date": doc.get("date", ""),
                                "source_name": doc.get("source_name", "로컬 문서"),
                            },
                            ensure_ascii=False,
                        )
                        cursor = conn.execute(
                            """
                            INSERT INTO rag_documents (source_path, title, url, date, source_name, metadata, content)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                doc.get("source_path", ""),
                                doc.get("title") or "문서",
                                doc.get("url", ""),
                                doc.get("date", ""),
                                doc.get("source_name", "로컬 문서"),
                                metadata,
                                doc.get("text") or doc.get("content") or "",
                            ),
                        )
                        conn.execute(
                            """
                            INSERT INTO rag_documents_fts (rowid, title, content, metadata)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                cursor.lastrowid,
                                doc.get("title") or "문서",
                                doc.get("text") or doc.get("content") or "",
                                metadata,
                            ),
                        )
        except Exception as exc:  # noqa: BLE001
            self._local_error = f"{type(exc).__name__}: {exc}"

    def _load_source_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for source_dir in self.source_dirs:
            if not source_dir.exists():
                continue
            for path in sorted(source_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".jsonl", ".json", ".txt", ".md"}:
                    continue
                documents.extend(self._load_source_file(path))
        return documents

    def _load_source_file(self, path: Path) -> list[dict[str, Any]]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        if path.suffix.lower() == ".jsonl":
            docs = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                docs.append(self._normalize_source_item(item, path))
            return docs
        if path.suffix.lower() == ".json":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return []
            items = parsed if isinstance(parsed, list) else [parsed]
            return [self._normalize_source_item(item, path) for item in items if isinstance(item, dict)]
        title = path.stem.replace("_", " ")
        return [
            {
                "source_path": str(path),
                "title": title,
                "url": "",
                "date": "",
                "source_name": "로컬 문서",
                "text": text.strip(),
            }
        ]

    @staticmethod
    def _normalize_source_item(item: dict[str, Any], path: Path) -> dict[str, Any]:
        text = item.get("text") or item.get("content") or item.get("body") or item.get("document") or ""
        return {
            "source_path": str(path),
            "title": str(item.get("title") or item.get("notice_title") or path.stem),
            "url": str(item.get("url") or item.get("notice_url") or item.get("source_url") or ""),
            "date": str(item.get("date") or item.get("notice_date") or ""),
            "source_name": str(item.get("source_name") or item.get("dept") or item.get("source") or "로컬 문서"),
            "text": str(text),
        }

    def _local_document_count(self) -> int:
        if not self.local_index_path.exists():
            return 0
        try:
            with sqlite3.connect(self.local_index_path) as conn:
                return int(conn.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0])
        except sqlite3.Error:
            return 0

    def _local_fts_search(self, question: str, top_k: int) -> list[SearchHit]:
        if not self.local_index_path.exists():
            return []
        query = _fts_query(question)
        try:
            with sqlite3.connect(self.local_index_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT d.title, d.url, d.date, d.source_name, d.content,
                           bm25(rag_documents_fts) AS score
                    FROM rag_documents_fts
                    JOIN rag_documents d ON d.id = rag_documents_fts.rowid
                    WHERE rag_documents_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (query, top_k),
                ).fetchall()
        except sqlite3.Error:
            return []

        hits: list[SearchHit] = []
        for row in rows:
            metadata = {
                "title": row["title"],
                "url": row["url"],
                "date": row["date"],
                "source_name": row["source_name"],
            }
            hits.append(
                SearchHit(
                    document=row["content"],
                    metadata=metadata,
                    score=float(row["score"]) if row["score"] is not None else None,
                )
            )
        return hits

    def _format_answer(self, hits: list[SearchHit], mode: str) -> str:
        lines = ["관련 자료를 찾았습니다.", ""]

        for number, hit in enumerate(hits, start=1):
            metadata = hit.metadata
            title = metadata.get("title") or metadata.get("notice_title") or f"관련 문서 {number}"
            source = metadata.get("source_name")
            date = metadata.get("date") or metadata.get("notice_date")
            url = metadata.get("url") or metadata.get("notice_url")
            heading = f"{number}. {title}"
            if date:
                heading += f" ({date})"
            if source:
                heading += f" - {source}"
            lines.append(heading)
            if url:
                lines.append(f"   출처: {url}")
            lines.append(f"   내용: {_clean_excerpt(hit.document)}")
            lines.append("")

        lines.append("공식 일정이나 모집/장학/학사 정보는 변경될 수 있으므로, 출처의 최신 공지를 함께 확인해 주세요.")
        return "\n".join(lines).strip()


_RAG_SERVICE: LocalRagService | None = None


def get_rag_service() -> LocalRagService:
    global _RAG_SERVICE
    if _RAG_SERVICE is None:
        _RAG_SERVICE = LocalRagService()
    return _RAG_SERVICE

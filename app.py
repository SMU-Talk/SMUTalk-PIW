from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from llm_service import generate_answer
from rag_service import get_rag_service


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

SESSION_COOKIE = "smu_chat_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 14
DEFAULT_SECRET = "dev-secret-change-me"


def load_env_file() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = os.environ.get("SMU_APP_NAME", "SMU Talk API")
    host: str = os.environ.get("SMU_CHAT_HOST", "127.0.0.1")
    port: int = int(os.environ.get("SMU_CHAT_PORT", "8000"))
    database_path: Path = Path(os.environ.get("SMU_DATABASE_PATH", str(DATA_DIR / "chatbot.sqlite3")))
    secret_key: str = os.environ.get("SMU_CHAT_SECRET", DEFAULT_SECRET)
    cookie_secure: bool = _env_bool("SMU_COOKIE_SECURE", False)
    frontend_origins: list[str] = None  # type: ignore[assignment]
    max_body_bytes: int = int(os.environ.get("SMU_MAX_BODY_BYTES", str(64 * 1024)))
    max_message_chars: int = int(os.environ.get("SMU_MAX_MESSAGE_CHARS", "2000"))
    max_history_messages: int = int(os.environ.get("SMU_MAX_HISTORY_MESSAGES", "100"))
    rate_limit_window_seconds: int = int(os.environ.get("SMU_RATE_LIMIT_WINDOW_SECONDS", "60"))
    rate_limit_default: int = int(os.environ.get("SMU_RATE_LIMIT_DEFAULT", "120"))
    rate_limit_login: int = int(os.environ.get("SMU_RATE_LIMIT_LOGIN", "12"))
    rate_limit_chat: int = int(os.environ.get("SMU_RATE_LIMIT_CHAT", "30"))
    allow_guest: bool = _env_bool("SMU_ALLOW_GUEST", True)
    enforce_strong_secret: bool = _env_bool("SMU_ENFORCE_STRONG_SECRET", False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frontend_origins",
            _env_list(
                "SMU_FRONTEND_ORIGINS",
                [
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://localhost:8000",
                    "http://127.0.0.1:8000",
                ],
            ),
        )


settings = Settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smu_chatbot")


def now() -> int:
    return int(time.time())


def db() -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.database_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                guest_name TEXT,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
            CREATE INDEX IF NOT EXISTS idx_messages_session_id_id ON messages(session_id, id);
            """
        )


def validate_runtime_config() -> None:
    if settings.secret_key == DEFAULT_SECRET:
        message = "SMU_CHAT_SECRET is using the development default."
        if settings.enforce_strong_secret:
            raise RuntimeError(f"{message} Set a long random secret before starting.")
        logger.warning("%s Set SMU_ENFORCE_STRONG_SECRET=true in production.", message)


def password_hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        210_000,
    ).hex()


def sign(value: str) -> str:
    digest = hmac.new(settings.secret_key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{value}.{digest}"


def unsign(value: str | None) -> str | None:
    if not value or "." not in value:
        return None
    session_id, signature = value.rsplit(".", 1)
    expected = sign(session_id).rsplit(".", 1)[1]
    if not hmac.compare_digest(signature, expected):
        return None
    return session_id


def clean_expired_sessions() -> None:
    with db() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now(),))


def create_session(user_id: int | None = None, guest_name: str | None = None) -> str:
    session_id = secrets.token_urlsafe(32)
    with db() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, user_id, guest_name, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_id, guest_name, now(), now() + SESSION_TTL_SECONDS),
        )
    return session_id


def get_session(session_id: str | None) -> sqlite3.Row | None:
    if not session_id:
        return None
    clean_expired_sessions()
    with db() as conn:
        return conn.execute(
            """
            SELECT sessions.*, users.username, users.display_name
            FROM sessions
            LEFT JOIN users ON users.id = sessions.user_id
            WHERE sessions.id = ?
            """,
            (session_id,),
        ).fetchone()


def current_session(request: Request) -> sqlite3.Row | None:
    return get_session(unsign(request.cookies.get(SESSION_COOKIE)))


def set_auth_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        sign(session_id),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/", samesite="lax", secure=settings.cookie_secure)


def api_error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_bytes:
            return api_error("요청 본문이 너무 큽니다.", 413)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        limit = self._limit_for(request.url.path)
        if limit <= 0:
            return await call_next(request)

        key = f"{request.client.host if request.client else 'unknown'}:{request.url.path}"
        bucket = self.requests[key]
        current = time.monotonic()
        window_start = current - settings.rate_limit_window_seconds
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= limit:
            return api_error("요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.", 429)
        bucket.append(current)
        return await call_next(request)

    @staticmethod
    def _limit_for(path: str) -> int:
        if path in {"/api/login", "/api/register", "/api/guest"}:
            return settings.rate_limit_login
        if path == "/api/chat":
            return settings.rate_limit_chat
        if path.startswith("/api/"):
            return settings.rate_limit_default
        return 0


FAQS = [
    {
        "keywords": ["입학", "수시", "정시", "전형", "모집"],
        "answer": "입학 관련 질문은 모집 시기, 전형 유형, 제출 서류가 핵심입니다. 실제 일정과 요강은 매년 바뀌므로 상명대학교 입학처 공지사항을 기준으로 확인하는 것이 가장 안전합니다.",
    },
    {
        "keywords": ["등록금", "장학", "장학금", "국가장학"],
        "answer": "등록금과 장학금은 학과, 학년, 장학 유형에 따라 달라집니다. 성적장학, 국가장학, 교내외 장학을 함께 확인하고, 신청 기간을 놓치지 않는 것이 중요합니다.",
    },
    {
        "keywords": ["학사", "수강", "수강신청", "휴학", "복학", "졸업"],
        "answer": "학사 업무는 학사 일정과 포털 공지가 우선입니다. 수강신청, 휴학, 복학, 졸업 요건은 소속 캠퍼스와 학과 기준이 다를 수 있어 학사 공지와 학과 사무실 안내를 함께 확인해 주세요.",
    },
    {
        "keywords": ["캠퍼스", "서울", "천안", "위치", "교통"],
        "answer": "상명대학교는 서울캠퍼스와 천안캠퍼스가 있습니다. 방문 목적에 따라 캠퍼스를 먼저 확인하고, 대중교통 또는 셔틀 안내를 함께 확인하면 이동 계획을 세우기 좋습니다.",
    },
    {
        "keywords": ["도서관", "열람실", "자료", "논문"],
        "answer": "도서관 이용은 자료 검색, 열람실, 전자자료, 논문 DB 이용으로 나눠 볼 수 있습니다. 로그인 권한이 필요한 서비스는 학교 계정 또는 도서관 인증 절차가 필요할 수 있습니다.",
    },
    {
        "keywords": ["포털", "샘물", "계정", "비밀번호", "로그인"],
        "answer": "학교 포털이나 샘물 시스템 로그인 문제는 계정 상태, 비밀번호, 브라우저 환경을 먼저 확인해 보세요. 해결되지 않으면 학교 IT 또는 행정 지원 창구로 문의하는 것이 빠릅니다.",
    },
]


def make_bot_reply(user_message: str, session: sqlite3.Row | None) -> str:
    text = user_message.strip()
    lowered = text.lower()
    display_name = "게스트"
    if session:
        display_name = session["display_name"] or session["guest_name"] or "게스트"

    if not text:
        return "질문을 입력해 주시면 상명대학교 생활, 입학, 학사, 장학, 캠퍼스 정보를 중심으로 도와드릴게요."

    rag_reply = get_rag_service().answer(text)
    if rag_reply:
        return generate_answer(text, rag_reply) or rag_reply

    if any(word in lowered for word in ["안녕", "hello", "hi"]):
        return f"안녕하세요, {display_name}님. SMU Talk입니다. 입학, 학사, 장학, 캠퍼스, 도서관, 포털 관련 질문을 편하게 입력해 주세요."

    for item in FAQS:
        if any(keyword in lowered for keyword in item["keywords"]):
            return item["answer"]

    return (
        "제가 바로 확정 답변을 드리기 어려운 질문입니다. "
        "질문을 '입학', '학사', '장학금', '캠퍼스', '도서관', '포털'처럼 주제와 함께 다시 적어주시면 더 정확히 안내할 수 있어요. "
        "공식 일정이나 규정은 최신 공지를 반드시 확인해 주세요."
    )


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=256)
    displayName: str = Field(default="", max_length=80)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class GuestRequest(BaseModel):
    guestName: str = Field(default="게스트", max_length=30)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_runtime_config()
    init_db()
    status = get_rag_service().status()
    logger.info("RAG enabled=%s vector_ready=%s path=%s", status["enabled"], status["vector_ready"], status["chroma_path"])
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=settings.max_body_bytes)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/full")
def health_full() -> dict[str, Any]:
    return {"status": "ok", "rag": get_rag_service().status()}


@app.get("/api/me")
def me(request: Request) -> dict[str, Any]:
    session = current_session(request)
    if not session:
        return {"authenticated": False, "mode": None, "name": None}
    mode = "user" if session["user_id"] else "guest"
    name = session["display_name"] if session["user_id"] else session["guest_name"]
    return {"authenticated": True, "mode": mode, "name": name}


@app.get("/api/history")
def history(request: Request) -> dict[str, Any]:
    session = current_session(request)
    if not session:
        return {"messages": []}
    with db() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (session["id"], settings.max_history_messages),
        ).fetchall()
    return {"messages": [dict(row) for row in rows]}


@app.get("/api/rag/status")
def rag_status() -> dict[str, Any]:
    return get_rag_service().status()


@app.post("/api/rag/reload")
def rag_reload() -> dict[str, Any]:
    return get_rag_service().reload()


@app.post("/api/register", response_model=None)
def register(payload: RegisterRequest, response: Response) -> Any:
    username = payload.username.strip().lower()
    password = payload.password
    display_name = payload.displayName.strip() or username

    if not username.replace("_", "").replace("-", "").isalnum():
        return api_error("아이디는 영문, 숫자, 하이픈, 밑줄만 사용할 수 있습니다.", 400)

    salt = secrets.token_hex(16)
    try:
        with db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, display_name, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, password_hash(password, salt), salt, display_name, now()),
            )
            user_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError:
        return api_error("이미 사용 중인 아이디입니다.", 409)

    set_auth_cookie(response, create_session(user_id=user_id))
    return {"ok": True, "name": display_name, "mode": "user"}


@app.post("/api/login", response_model=None)
def login(payload: LoginRequest, response: Response) -> Any:
    username = payload.username.strip().lower()
    password = payload.password

    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if not user or not hmac.compare_digest(user["password_hash"], password_hash(password, user["salt"])):
        return api_error("아이디 또는 비밀번호를 확인해 주세요.", 401)

    set_auth_cookie(response, create_session(user_id=user["id"]))
    return {"ok": True, "name": user["display_name"], "mode": "user"}


@app.post("/api/guest", response_model=None)
def guest(payload: GuestRequest, response: Response) -> Any:
    if not settings.allow_guest:
        return api_error("게스트 모드가 비활성화되어 있습니다.", 403)
    guest_name = payload.guestName.strip()[:30] or "게스트"
    set_auth_cookie(response, create_session(guest_name=guest_name))
    return {"ok": True, "name": guest_name, "mode": "guest"}


@app.post("/api/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    session = current_session(request)
    if session:
        with db() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session["id"],))
    clear_auth_cookie(response)
    return {"ok": True}


@app.post("/api/clear")
def clear(request: Request) -> dict[str, bool]:
    session = current_session(request)
    if session:
        with db() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session["id"],))
    return {"ok": True}


@app.post("/api/chat", response_model=None)
def chat(payload: ChatRequest, request: Request) -> Any:
    session = current_session(request)
    if not session:
        return api_error("로그인 또는 게스트 모드로 입장해 주세요.", 401)

    message = payload.message.strip()
    if not message:
        return api_error("메시지를 입력해 주세요.", 400)
    if len(message) > settings.max_message_chars:
        return api_error(f"메시지는 {settings.max_message_chars}자 이하로 입력해 주세요.", 400)

    reply = make_bot_reply(message, session)
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
            (session["id"], message, now()),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)",
            (session["id"], reply, now()),
        )
    return {"reply": reply}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def run() -> None:
    import uvicorn

    uvicorn.run("app:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()

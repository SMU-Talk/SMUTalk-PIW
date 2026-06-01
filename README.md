# SMU Talk

상명대학교 공지 데이터를 활용하는  FastAPI 기반 챗봇 서비스입니다. 
## 주요 변경점

- FastAPI + Uvicorn 기반 API 서버
- 안전한 정적 파일 서빙
- `/health`, `/health/full` 운영 점검 API
- CORS 허용 origin 환경변수화
- 로그인/게스트/채팅 API rate limit
- 운영용 세션 쿠키 옵션(`HttpOnly`, `SameSite=Lax`, 선택적 `Secure`)
- 기본 개발 시크릿 사용 경고 및 운영 시 강제 실패 옵션
- SQLite WAL, 외래키, 인덱스 적용

## 설치

Python 3.11 권장.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 실행

개발 실행:

```powershell
python -X utf8 app.py
```

또는:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`으로 접속합니다.

## OpenAI API 답변 생성

기본 동작은 RAG 검색 결과를 정해진 형식으로 보여주는 방식입니다. OpenAI API 키를 설정하면 검색 결과를 바탕으로 더 자연스러운 한국어 답변을 생성합니다.

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:SMU_USE_OPENAI="true"
$env:SMU_OPENAI_MODEL="gpt-5.4-mini"
python -X utf8 app.py
```

`.env` 파일에 넣어도 됩니다.

```env
OPENAI_API_KEY=sk-...
SMU_USE_OPENAI=true
SMU_OPENAI_MODEL=gpt-5.4-mini
SMU_OPENAI_MAX_CONTEXT_CHARS=9000
SMU_OPENAI_TIMEOUT_SECONDS=20
```

API 키가 없거나 OpenAI 호출이 실패하면 기존 RAG 검색 결과로 자동 fallback합니다. API 키는 브라우저 코드에 넣지 말고 서버 환경변수로만 설정하세요.

주의: 이 연동은 OpenAI가 웹을 실시간 검색하는 기능이 아니라, 서버가 찾은 RAG 검색 결과를 요약하는 기능입니다. 학교 홈페이지의 정적 안내 페이지까지 정확히 답하려면 해당 페이지 내용을 `rag_sources/`에 넣거나 별도 크롤러로 인덱싱해야 합니다.

## 운영 환경변수

`.env.example`을 참고해 배포 환경에 맞는 환경변수를 설정합니다. 운영에서는 최소한 아래 값은 반드시 바꾸세요.

```powershell
$env:SMU_CHAT_SECRET="긴-랜덤-문자열"
$env:SMU_ENFORCE_STRONG_SECRET="true"
$env:SMU_FRONTEND_ORIGINS="https://your-frontend.example.com"
$env:SMU_COOKIE_SECURE="true"
```

`SMU_ENFORCE_STRONG_SECRET=true`일 때 `SMU_CHAT_SECRET`이 기본값이면 서버가 시작되지 않습니다.

## Chroma 데이터 연결

앱은 다음 순서로 RAG 검색을 시도합니다.

1. Qwen3-Embedding-4B E1 artifact (`embeddings.npy` + `chunks.jsonl`)
2. Chroma DB
3. `rag_sources/` 로컬 문서 SQLite FTS

## Qwen3-Embedding-4B E1 Artifact 연결

`smu_notice_qwen3_e1_release` 산출물은 Chroma DB가 아니라 `embeddings.npy`와 `chunks.jsonl`로 구성됩니다. 프로젝트 루트에 아래처럼 배치하면 자동으로 인식합니다.

```text
qwen3_e1/
├─ embeddings.npy
├─ chunks.jsonl
├─ manifest.json
├─ README.md
└─ ...
```

또는 환경변수로 경로를 지정합니다.

```powershell
$env:SMU_E1_ARTIFACT_DIR="C:\path\to\smu_notice_qwen3_e1_release"
```

현재 구현은 두 단계로 동작합니다.

- `SMU_E1_MODEL_DIR`가 있거나 `SMU_E1_ALLOW_MODEL_DOWNLOAD=true`이면 Qwen3-Embedding-4B 쿼리 벡터 검색을 시도합니다.
- 모델이 없으면 `chunks.jsonl` 전체를 SQLite FTS로 인덱싱해 `Qwen3 E1 문서 검색` 모드로 검색합니다.

Qwen3-Embedding-4B 모델을 로컬에 내려받은 경우:

```powershell
$env:SMU_E1_MODEL_DIR="C:\models\Qwen3-Embedding-4B"
$env:SMU_E1_ENABLE_VECTOR="true"
python -X utf8 app.py
```

모델 자동 다운로드를 허용하려면:

```powershell
$env:SMU_E1_ALLOW_MODEL_DOWNLOAD="true"
```

주의: Qwen3-Embedding-4B는 메모리와 디스크 사용량이 큽니다. AWS 배포 시 CPU 인스턴스에서는 느릴 수 있고, GPU 인스턴스나 별도 임베딩 서버 구성을 권장합니다.

## Chroma 데이터 연결

Chroma DB가 있으면 벡터 검색을 우선 사용하고, 실패하면 Chroma SQLite 전문검색으로 fallback합니다.

Chroma DB 폴더는 다음 순서로 찾습니다.

1. `SMU_CHROMA_PATH` 환경 변수
2. 프로젝트의 `vector_db/` 폴더
3. 다운로드 폴더의 `chroma_qwen3_embedding_0_6b_smu_notices_crawled_*` 폴더

로컬 RAG 문서는 기본적으로 `rag_sources/`, `notices/`, `data/rag_sources/`에서 읽습니다. `.jsonl`, `.json`, `.txt`, `.md` 파일을 지원합니다.

JSONL 예시:

```json
{"title":"공지 제목","date":"2026-06-01","source_name":"학사공지","url":"https://example.com","text":"검색에 사용할 본문"}
```

문서를 추가한 뒤 서버 재시작 없이 다시 인덱싱하려면:

```powershell
Invoke-WebRequest -UseBasicParsing -Method POST http://127.0.0.1:8000/api/rag/reload
```

상태 확인:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/rag/status
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health/full
```



## 배포 참고

- `SMU_CHAT_HOST=0.0.0.0`은 HTTPS reverse proxy 뒤에서만 권장합니다.
- HTTPS 배포 시 `SMU_COOKIE_SECURE=true`를 설정하세요.
- 게스트 사용을 막으려면 `SMU_ALLOW_GUEST=false`를 설정하세요.
- SQLite는 소규모 서비스에 적합합니다. 동시 사용자가 늘면 PostgreSQL 등 서버형 DB로 이전하는 것이 좋습니다.
- `/health/full`은 RAG 경로와 오류 정보를 노출할 수 있으므로 운영 환경에서는 방화벽 또는 프록시에서 접근 제한을 고려하세요.

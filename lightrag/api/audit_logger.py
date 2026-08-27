"""
MySQL-based audit logging for LightRAG's query endpoints.
 
Same schema and function signatures as the previous SQLite version —
query_routes.py and session_routes.py do not need to change, aside from
importing get_connection() here instead of sqlite3.connect(DB_PATH)
directly (see session_routes.py).
 
Requires: uv add pymysql
 
Configuration via env vars (set in .env):
    MYSQL_HOST      default: localhost
    MYSQL_PORT      default: 3306
    MYSQL_USER      required
    MYSQL_PASSWORD  required
    MYSQL_DATABASE  default: lightrag_audit
 
This module assumes the database itself already exists — creating it
requires a privilege (CREATE) you may not want to grant the app's MySQL
user in a bank environment. Create it once, out of band:
 
    CREATE DATABASE IF NOT EXISTS lightrag_audit
        CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
 
Tables inside that database are still self-healing, same as before —
every write ensures its table exists first, so no separate init step
is required for day-to-day use.
 
Drop this file at: lightrag/api/audit_logger.py
"""
 
import os
import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List
 
import pymysql
import pymysql.cursors
from dotenv import load_dotenv
 
# Same convention as auth.py: load .env from the current working directory,
# without overriding real OS environment variables if already set. Without
# this, MYSQL_USER/MYSQL_PASSWORD/etc. below silently read as None whenever
# nothing earlier in the import chain has already loaded .env — pymysql then
# falls back to your OS username with no password, which fails confusingly.
load_dotenv(dotenv_path=".env", override=False)
 
# MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
# MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
# MYSQL_USER = os.getenv("MYSQL_USER")
# MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
# MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "lightrag_audit")
 
# MYSQL_HOST = "192.168.1.170"
# MYSQL_PORT = 3306
# MYSQL_USER = "sahild"
# MYSQL_PASSWORD = "Viking@@ibs2026"
# MYSQL_DATABASE = "unity_connect"
 
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "lightrag_audit")
 
_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS query_logs (
        sr_no           BIGINT AUTO_INCREMENT PRIMARY KEY,
        id              VARCHAR(36) NOT NULL UNIQUE,
        user_id         VARCHAR(255),
        thread_id       VARCHAR(255),
        thread_msg_no   INT,
        department      VARCHAR(255),
        user_query      TEXT,
        llm_response    LONGTEXT,
        status          VARCHAR(20) NOT NULL,
        error_message   TEXT,
        citations       TEXT,
        created_at      DATETIME(6) NOT NULL,
        INDEX idx_query_logs_user_id (user_id),
        INDEX idx_query_logs_thread_id (thread_id),
        INDEX idx_query_logs_thread_msg (thread_id, thread_msg_no),
        INDEX idx_query_logs_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS thread_counters (
        thread_id   VARCHAR(255) PRIMARY KEY,
        turn_count  BIGINT NOT NULL DEFAULT 0,
        updated_at  DATETIME(6) NOT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS user_rbac (
        user_id     VARCHAR(255) PRIMARY KEY,
        department  VARCHAR(255) NOT NULL,
        role        VARCHAR(50),
        source      VARCHAR(20) NOT NULL DEFAULT 'manual',
        updated_at  DATETIME(6) NOT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS guardrail_logs (
        sr_no                   BIGINT AUTO_INCREMENT PRIMARY KEY,
        id                      VARCHAR(36) NOT NULL UNIQUE,
        query_log_id            VARCHAR(36),
        user_id                 VARCHAR(255),
        thread_id               VARCHAR(255),
        department              VARCHAR(255),
        user_query              TEXT,
        llm_response            LONGTEXT,
        input_guardrail_status  VARCHAR(10) NOT NULL,
        input_guardrail_reason  TEXT,
        input_guardrail_details JSON,
        output_guardrail_status  VARCHAR(10) NOT NULL DEFAULT 'not_run',
        output_guardrail_reason  TEXT,
        output_guardrail_details JSON,
        created_at              DATETIME(6) NOT NULL,
        updated_at              DATETIME(6) NOT NULL,
        INDEX idx_guardrail_logs_query_log_id (query_log_id),
        INDEX idx_guardrail_logs_user_id (user_id),
        INDEX idx_guardrail_logs_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
     """
    CREATE TABLE IF NOT EXISTS private_chunk_access_logs (
        id              VARCHAR(36) NOT NULL UNIQUE,
        user_id         VARCHAR(255),
        thread_id       VARCHAR(255),
        department      VARCHAR(255),
        user_query      TEXT,
        total_chunks    INT UNSIGNED NOT NULL,
        private_chunks  INT UNSIGNED NOT NULL,
        private_pct     DECIMAL(5,2) NOT NULL,
        private_files   JSON,
        blocked_chunk_ids JSON,
        created_at      DATETIME(6) NOT NULL,
        INDEX idx_pca_user_id    (user_id),
        INDEX idx_pca_created_at (created_at),
        INDEX idx_pca_pct        (private_pct)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
 
]
 
 
def get_connection():
    """Open a new MySQL connection. Exposed for reuse by other custom
    routers (e.g. session_routes.py) that need direct read access to
    the same tables."""
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
        autocommit=False,
    )
 
 
def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        for stmt in _SCHEMA_STATEMENTS:
            cur.execute(stmt)
    conn.commit()
 
 
# Public alias — other custom modules (e.g. guardrails.py) that reuse
# get_connection() should call this rather than the underscored name.
ensure_schema = _ensure_schema
 
 
def _init_db_sync() -> None:
    conn = get_connection()
    try:
        _ensure_schema(conn)
    finally:
        conn.close()
 
 
def _next_thread_msg_no_sync(conn, thread_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO thread_counters (thread_id, turn_count, updated_at)
            VALUES (%s, LAST_INSERT_ID(1), %s)
            ON DUPLICATE KEY UPDATE
                turn_count = LAST_INSERT_ID(turn_count + 1),
                updated_at = VALUES(updated_at)
            """,
            (thread_id, datetime.now(timezone.utc).replace(tzinfo=None)),
        )
        cur.execute("SELECT LAST_INSERT_ID()")
        return cur.fetchone()[0]
 
 
def _insert_sync(row: dict) -> None:
    conn = get_connection()
    try:
        _ensure_schema(conn)
 
        thread_msg_no = None
        if row.get("thread_id"):
            thread_msg_no = _next_thread_msg_no_sync(conn, row["thread_id"])
 
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_logs (
                    id, user_id, thread_id, thread_msg_no, department,
                    user_query, llm_response, status, error_message,
                    citations, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row["id"],
                    row.get("user_id"),
                    row.get("thread_id"),
                    thread_msg_no,
                    row.get("department"),
                    row.get("user_query"),
                    row.get("llm_response"),
                    row["status"],
                    row.get("error_message"),
                    row.get("citations"),
                    row["created_at"],
                ),
            )
        conn.commit()
    finally:
        conn.close()
       
 
async def init_audit_db() -> None:
    """Optional: call once at server startup to pre-create tables. Not
    required — every write self-heals via _ensure_schema()."""
    await asyncio.to_thread(_init_db_sync)
 
 
async def log_query_event(
    *,
    log_id: Optional[str] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    department: Optional[str] = None,
    user_query: Optional[str] = None,
    llm_response: Optional[str] = None,
    status: str = "success",  # "success" | "fallback" | "error"
    error_message: Optional[str] = None,
    citations: Optional[List[str]] = None,
) -> str:
    """Insert one audit log row. Returns the generated id (UUID).
 
    Runs the actual MySQL write in a worker thread via asyncio.to_thread
    (pymysql is synchronous), so it never blocks the event loop. Failures
    to write the audit log are caught and sent to the normal LightRAG
    logger instead of raising — audit logging should never be able to
    break a real user request.
    """
    # log_id = str(uuid.uuid4())
    row = {
        "id": log_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "department": department,
        "user_query": user_query,
        "llm_response": llm_response,
        "status": status,
        "error_message": error_message,
        "citations": json.dumps(citations or []),
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
    try:
        await asyncio.to_thread(_insert_sync, row)
    except Exception:
        from lightrag.utils import logger as _logger
 
        _logger.error("Failed to write audit log row", exc_info=True)
    return log_id
 
 
def _lookup_department_sync(user_id: str) -> Optional[str]:
    conn = get_connection()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT department FROM user_rbac WHERE user_id = %s", (user_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()
 
 
async def get_department(user_id: Optional[str]) -> Optional[str]:
    """Look up a user's department from the local user_rbac table.
 
    TODO: once the bank's RBAC endpoint is available, replace (or
    supplement) this with a call to that API — either on every lookup
    with a short in-memory cache, or via a periodic sync job that keeps
    upsert_user_rbac() populated in the background. Nothing else in the
    audit pipeline needs to change; this is the single choke point.
    """
    if not user_id:
        return None
    try:
        return await asyncio.to_thread(_lookup_department_sync, user_id)
    except Exception:
        from lightrag.utils import logger as _logger
 
        _logger.error("Failed to look up department for audit log", exc_info=True)
        return None
 
 
def _upsert_rbac_sync(user_id: str, department: str, role: Optional[str], source: str) -> None:
    conn = get_connection()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_rbac (user_id, department, role, source, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    department = VALUES(department),
                    role = VALUES(role),
                    source = VALUES(source),
                    updated_at = VALUES(updated_at)
                """,
                (
                    user_id,
                    department,
                    role,
                    source,
                    datetime.now(timezone.utc).replace(tzinfo=None),
                ),
            )
        conn.commit()
    finally:
        conn.close()
 
 
async def upsert_user_rbac(
    user_id: str,
    department: str,
    role: Optional[str] = None,
    source: str = "manual",
) -> None:
    """Seed or update one user's department mapping.
 
    Use this manually for now (e.g. a small one-off script) to get real
    department values into the logs while the bank's endpoint isn't
    available yet. Once that endpoint exists, a background sync task can
    call this same function per user on a schedule, passing
    source="bank_api" instead of "manual" so you can tell the two apart.
    """
    await asyncio.to_thread(_upsert_rbac_sync, user_id, department, role, source)
 
 
async def extract_user_context(http_request) -> tuple[Optional[str], Optional[str]]:
    """Best-effort extraction of (user_id, department) from a FastAPI
    Request, using the same JWT the real auth dependency already validated.
 
    This re-decodes the Bearer token via auth_handler.validate_token() purely
    to read identity claims for logging — it never blocks the request.
    combined_auth (Depends) already made the real access-control decision
    before this route handler runs; any failure here (missing/invalid token,
    e.g. API-key-only or guest access) just means user_id stays None rather
    than raising.
 
    department resolution order:
      1. The JWT's metadata.department claim, if you're setting one there.
      2. The local user_rbac table (see upsert_user_rbac / get_department) —
         this is the interim source until the bank's RBAC endpoint exists.
      3. The X-Department header, as a last-resort manual override.
    """
    user_id: Optional[str] = None
    department: Optional[str] = None
 
    auth_header = http_request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer ") :].strip()
        try:
            from lightrag.api.auth import auth_handler
 
            payload = auth_handler.validate_token(token)
            user_id = payload.get("username")
            department = (payload.get("metadata") or {}).get("department")
        except Exception:
            # Invalid/expired token, guest token, or API-key-only auth mode
            # (no Bearer token at all reaches here). Not an audit-logger
            # failure — just no identity to attach.
            pass
 
    # Fallback for user_id: explicit header, useful for API-key-only
    # deployments where there's no JWT to decode, or for testing.
    if not user_id:
        user_id = http_request.headers.get("X-User-Id")
 
    # department fallback chain: local RBAC table, then header override.
    if not department and user_id:
        department = await get_department(user_id)
    if not department:
        department = http_request.headers.get("X-Department")
 
    return user_id, department
 
 
def extract_thread_id(http_request) -> Optional[str]:
    """Reads the conversation/thread id generated by the frontend Node
    server. Expected as an 'X-Thread-Id' request header.
 
    If your frontend instead sends this as a query-body field rather than
    a header, read it from the parsed request body at the call site
    instead — this helper only covers the header convention.
    """
    return http_request.headers.get("X-Thread-Id") or http_request.headers.get(
        "X-Thread-ID"
    )
 
 
def extract_citations(references: Optional[List[dict]]) -> List[str]:
    """Turns a LightRAG references list (each item has a 'file_path' key)
    into a deduplicated list of source file names, in first-seen order.
 
    e.g. [{"file_path": "/documents/Deposit Policy.pdf"}, ...]
      -> ["Deposit Policy.pdf"]
    """
    if not references:
        return []
    seen = []
    for ref in references:
        file_path = ref.get("file_path") if isinstance(ref, dict) else None
        if not file_path:
            continue
        name = os.path.basename(file_path)
        if name not in seen:
            seen.append(name)
    return seen
 
 
def _insert_private_chunk_log_sync(row: dict) -> None:
    conn = get_connection()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO private_chunk_access_logs (
                    id, user_id, thread_id, department,
                    user_query, total_chunks, private_chunks,
                    private_pct, private_files, blocked_chunk_ids, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row["id"],
                    row.get("user_id"),
                    row.get("thread_id"),
                    row.get("department"),
                    row.get("user_query"),
                    row["total_chunks"],
                    row["private_chunks"],
                    row["private_pct"],
                    json.dumps(row.get("private_files") or []),
                    json.dumps(row.get("blocked_chunk_ids") or []),
                    row["created_at"],
                ),
            )
        conn.commit()
    finally:
        conn.close()
 
 
async def log_private_chunk_access(
    *,
    id: Optional[str] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    department: Optional[str] = None,
    user_query: str,
    total_chunks: int,
    private_chunks: int,
    private_files: Optional[List[str]] = None,
    blocked_chunk_ids: Optional[List[str]] = None,
) -> None:
    """
    Logs to private_chunk_access_logs when >40% of retrieved chunks
    came from private documents, for user_type == 'user'.
 
    Mirrors the fire-and-forget pattern of log_query_event:
    never raises, never blocks the event loop.
    """
    private_pct = round((private_chunks / total_chunks) * 100, 2)
    row = {
        "id": id,
        "user_id": user_id,
        "thread_id": thread_id,
        "department": department,
        "user_query": user_query,
        "total_chunks": total_chunks,
        "private_chunks": private_chunks,
        "private_pct": private_pct,
        "private_files": private_files or [],
        "blocked_chunk_ids": blocked_chunk_ids or [],
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
    try:
        await asyncio.to_thread(_insert_private_chunk_log_sync, row)
    except Exception:
        from lightrag.utils import logger as _logger
        _logger.error("Failed to write private_chunk_access_logs row", exc_info=True)
 
 
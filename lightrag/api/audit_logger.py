"""
MySQL-based audit logging for LightRAG's query endpoints.

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

Tables inside that database are still self-healing — every write ensures
its table exists first, so no separate init step is required.

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

load_dotenv(dotenv_path=".env", override=False)

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "lightrag_audit")

_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS query_logs (
        query_id        VARCHAR(36) NOT NULL PRIMARY KEY,
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
        query_log_id             VARCHAR(36) NOT NULL PRIMARY KEY,
        input_guardrail_status   VARCHAR(10) NOT NULL,
        input_guardrail_reason   TEXT,
        input_guardrail_details  JSON,
        output_guardrail_status  VARCHAR(10) NOT NULL DEFAULT 'not_run',
        output_guardrail_reason  TEXT,
        output_guardrail_details JSON,
        created_at               DATETIME(6) NOT NULL,
        updated_at               DATETIME(6) NOT NULL,
        INDEX idx_guardrail_logs_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS private_chunk_access_logs (
        query_id            VARCHAR(36) NOT NULL PRIMARY KEY,
        total_chunks        INT UNSIGNED NOT NULL,
        private_chunks      INT UNSIGNED NOT NULL,
        private_pct         DECIMAL(5,2) NOT NULL,
        private_files       JSON,
        blocked_chunk_ids   JSON,
        created_at          DATETIME(6) NOT NULL,
        INDEX idx_pca_created_at (created_at),
        INDEX idx_pca_pct        (private_pct)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def get_connection():
    """Open a new MySQL connection. Exposed for reuse by other custom
    routers that need direct read access to the same tables."""
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
    """Atomically increments and returns the message number for a thread_id.

    Uses MySQL's INSERT ... ON DUPLICATE KEY UPDATE ... LAST_INSERT_ID()
    trick: a single atomic statement, so two concurrent requests for the
    same thread_id can never receive the same number, unlike a naive
    SELECT MAX(thread_msg_no) + 1 (a classic race condition).

    Both branches wrap their value in LAST_INSERT_ID(...) explicitly —
    the very first insert must do this too, or SELECT LAST_INSERT_ID()
    right after returns whatever it was previously (often 0) instead of
    the 1 we just wrote, producing the "0, then jumps to 2" bug.
    """
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
        _ensure_schema(conn)  # self-healing, same behavior as before

        thread_msg_no = None
        if row.get("thread_id"):
            thread_msg_no = _next_thread_msg_no_sync(conn, row["thread_id"])

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_logs (
                    query_id, user_id, thread_id, thread_msg_no, department,
                    user_query, llm_response, status, error_message,
                    citations, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row["query_id"],
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
    query_id: str,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    department: Optional[str] = None,
    user_query: Optional[str] = None,
    llm_response: Optional[str] = None,
    status: str = "success",  # "success" | "fallback" | "error"
    error_message: Optional[str] = None,
    citations: Optional[List[str]] = None,
) -> str:
    """Insert one audit log row into query_logs, keyed by ``query_id``.

    ``query_id`` MUST be the same UUID passed to ``log_input_guardrail``
    for this request (generated once per request by the caller), so that
    ``query_logs.query_id`` and ``guardrail_logs.query_log_id`` always
    refer to the same query — no more separately-generated ids per table.

    Runs the actual MySQL write in a worker thread via asyncio.to_thread
    (pymysql is synchronous), so it never blocks the event loop. Failures
    to write the audit log are caught and sent to the normal LightRAG
    logger instead of raising — audit logging should never be able to
    break a real user request.
    """
    row = {
        "query_id": query_id,
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
    return query_id


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
    supplement) this with a call to that API.
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
    """Seed or update one user's department mapping."""
    await asyncio.to_thread(_upsert_rbac_sync, user_id, department, role, source)


async def extract_user_context(http_request) -> tuple[Optional[str], Optional[str]]:
    """Best-effort extraction of (user_id, department) from a FastAPI
    Request, using the same JWT the real auth dependency already validated.

    department resolution order:
      1. The JWT's metadata.department claim, if you're setting one there.
      2. The local user_rbac table.
      3. The X-Department header, as a last-resort manual override.

    Note: this only reads headers/JWT, never the parsed request body. If
    your frontend sends user_id/department as body fields instead, read
    those directly off your request model at the call site and prefer
    them over this function's return value (same pattern as thread_id).
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
            pass

    if not user_id:
        user_id = http_request.headers.get("X-User-Id")

    if not department and user_id:
        department = await get_department(user_id)
    if not department:
        department = http_request.headers.get("X-Department")

    return user_id, department


def extract_thread_id(http_request) -> Optional[str]:
    """Reads the conversation/thread id from the 'X-Thread-Id' header.

    If your frontend instead sends thread_id as a request-body field,
    prefer that value at the call site and use this only as a fallback —
    e.g. `thread_id = request.thread_id or extract_thread_id(http_request)`.
    """
    return http_request.headers.get("X-Thread-Id") or http_request.headers.get(
        "X-Thread-ID"
    )


def _insert_private_chunk_access_sync(row: dict) -> None:
    conn = get_connection()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO private_chunk_access_logs (
                    query_id, total_chunks, private_chunks, private_pct,
                    private_files, blocked_chunk_ids, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row["query_id"],
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
    query_id: str,
    total_chunks: int,
    private_chunks: int,
    private_files: Optional[List[str]] = None,
    blocked_chunk_ids: Optional[List[str]] = None,
) -> None:
    """Insert one private_chunk_access_logs row, keyed by ``query_id`` — the
    SAME uuid shared with query_logs.query_id and guardrail_logs.query_log_id
    for this request (generate it once via new_query_id() and pass it to
    every logging call for that request).

    user_id/thread_id/department/user_query are deliberately not stored
    here — they already live on the query_logs row for this query_id;
    join on query_id when you need them alongside these access-pattern
    stats. private_pct is computed here so callers never have to repeat
    that arithmetic (and risk divide-by-zero) at every call site.
    """
    private_pct = round((private_chunks / total_chunks) * 100, 2) if total_chunks else 0.0
    row = {
        "query_id": query_id,
        "total_chunks": total_chunks,
        "private_chunks": private_chunks,
        "private_pct": private_pct,
        "private_files": private_files or [],
        "blocked_chunk_ids": blocked_chunk_ids or [],
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
    try:
        await asyncio.to_thread(_insert_private_chunk_access_sync, row)
    except Exception:
        from lightrag.utils import logger as _logger

        _logger.error("Failed to write private chunk access log row", exc_info=True)


def new_query_id() -> str:
    """Generates the single UUID shared by query_logs.query_id and
    guardrail_logs.query_log_id for one request. Call this ONCE per
    request, at the top of the route, before calling log_input_guardrail
    or log_query_event — then pass the same value to both."""
    return str(uuid.uuid4())


def extract_citations(references: Optional[List[dict]]) -> List[str]:
    """Turns a LightRAG references list (each item has a 'file_path' key)
    into a deduplicated list of source file names, in first-seen order.
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
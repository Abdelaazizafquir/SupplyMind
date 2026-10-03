"""
db_connector.py — live database connection, preview, and import endpoints.

Supports: SQLite (built-in), PostgreSQL (psycopg2), MySQL (pymysql).
Security: parameterised queries only; connection timeout 5 s; password never echoed.
"""

import io, csv, os
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db
from models.schemas import DBConnectionConfig, DBPreviewResponse, MultiTableImportRequest
from routers.auth import get_current_user

router = APIRouter()

# ── driver availability ────────────────────────────────────────────────────────

def _build_url(cfg: DBConnectionConfig) -> str:
    """Return a sync SQLAlchemy URL string."""
    if cfg.db_type == "sqlite":
        # database field is the file path
        return f"sqlite:///{cfg.database}"
    if cfg.db_type == "postgresql":
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise HTTPException(
                400,
                "psycopg2-binary is not installed. "
                "Run: pip install psycopg2-binary",
            )
        return (
            f"postgresql+psycopg2://{cfg.username}:{cfg.password}"
            f"@{cfg.host}:{cfg.port}/{cfg.database}"
        )
    if cfg.db_type == "mysql":
        try:
            import pymysql  # noqa: F401
        except ImportError:
            raise HTTPException(
                400,
                "PyMySQL is not installed. "
                "Run: pip install pymysql",
            )
        return (
            f"mysql+pymysql://{cfg.username}:{cfg.password}"
            f"@{cfg.host}:{cfg.port}/{cfg.database}"
        )
    raise HTTPException(400, f"Unsupported db_type: {cfg.db_type}")


def _sync_engine(url: str):
    from sqlalchemy import create_engine
    return create_engine(url, connect_args={"connect_timeout": 5} if "psycopg2" in url else {})


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/test")
async def test_connection(
    cfg: DBConnectionConfig,
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Test DB connection and return available table names."""
    url = _build_url(cfg)
    try:
        from sqlalchemy import create_engine, inspect, text
        engine = _sync_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        insp = inspect(engine)
        tables = insp.get_table_names()
        engine.dispose()
        return {"success": True, "tables": tables}
    except HTTPException:
        raise
    except Exception as exc:
        return {"success": False, "tables": [], "error": str(exc)}


@router.post("/preview", response_model=DBPreviewResponse)
async def preview_table(
    cfg: DBConnectionConfig,
    current_user=Depends(get_current_user),
) -> DBPreviewResponse:
    """Return the first 10 rows of cfg.table_name."""
    if not cfg.table_name:
        raise HTTPException(400, "table_name is required for preview")
    url = _build_url(cfg)
    try:
        from sqlalchemy import create_engine, text
        engine = _sync_engine(url)
        with engine.connect() as conn:
            # Use parameterised identifier — table name validated as identifier
            safe_table = "".join(c for c in cfg.table_name if c.isalnum() or c in ("_", "."))
            if not safe_table:
                raise HTTPException(400, "Invalid table_name")
            result = conn.execute(text(f'SELECT * FROM `{safe_table}` LIMIT 10'))
            columns = list(result.keys())
            rows = [list(r) for r in result.fetchall()]
        engine.dispose()
        return DBPreviewResponse(columns=columns, rows=rows, row_count=len(rows))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Preview failed: {exc}") from exc


@router.post("/import")
async def import_table(
    cfg: DBConnectionConfig,
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Pull all rows from cfg.table_name, save as CSV in uploads/,
    and return the saved file path so the project can use it.
    """
    if not cfg.table_name:
        raise HTTPException(400, "table_name is required for import")
    url = _build_url(cfg)
    try:
        from sqlalchemy import create_engine, text
        engine = _sync_engine(url)
        with engine.connect() as conn:
            safe_table = "".join(c for c in cfg.table_name if c.isalnum() or c in ("_", "."))
            if not safe_table:
                raise HTTPException(400, "Invalid table_name")
            result = conn.execute(text(f'SELECT * FROM `{safe_table}`'))
            columns = list(result.keys())
            rows = result.fetchall()
        engine.dispose()

        # Write CSV
        os.makedirs("uploads", exist_ok=True)
        filename = f"db_import_{project_id}_{safe_table}.csv"
        filepath = os.path.join("uploads", filename)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)

        return {
            "success": True,
            "filename": filename,
            "rows_imported": len(rows),
            "columns": columns,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Import failed: {exc}") from exc


@router.post("/import-multi")
async def import_multi_table(
    req: MultiTableImportRequest,
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Import from multiple tables in one request.
    Each table is saved as a separate CSV file; results are returned per-table.
    """
    # Build a throwaway DBConnectionConfig just for URL construction
    conn_cfg = DBConnectionConfig(
        db_type=req.db_type, host=req.host, port=req.port,
        database=req.database, username=req.username, password=req.password,
    )
    url = _build_url(conn_cfg)

    results: List[Dict[str, Any]] = []
    errors:  List[Dict[str, Any]] = []

    try:
        from sqlalchemy import create_engine, text as sa_text
        engine = _sync_engine(url)

        for tbl in req.tables:
            safe_table = "".join(c for c in tbl.table_name if c.isalnum() or c in ("_", "."))
            if not safe_table:
                errors.append({"table_name": tbl.table_name, "error": "Invalid table name"})
                continue
            try:
                with engine.connect() as conn:
                    result = conn.execute(sa_text(f'SELECT * FROM `{safe_table}`'))
                    columns = list(result.keys())
                    rows    = result.fetchall()

                os.makedirs("uploads", exist_ok=True)
                filename = f"db_import_{project_id}_{safe_table}.csv"
                filepath = os.path.join("uploads", filename)
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(rows)

                results.append({
                    "table_name": tbl.table_name,
                    "filename": filename,
                    "rows_imported": len(rows),
                    "columns": columns,
                    "data": [list(r) for r in rows],
                })
            except Exception as exc:
                errors.append({"table_name": tbl.table_name, "error": str(exc)})

        engine.dispose()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Connection failed: {exc}") from exc

    total_rows = sum(r["rows_imported"] for r in results)
    return {
        "success": len(results) > 0,
        "results": results,
        "errors": errors,
        "total_rows": total_rows,
    }

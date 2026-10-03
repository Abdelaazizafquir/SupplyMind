"""Upload router — CSV/Excel data ingestion + EDA summary"""
import uuid
import os
import json
from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import pandas as pd
import aiofiles

from models.database import Dataset, get_db, User
from models.schemas import DatasetOut, ColumnMapping
from routers.auth import get_current_user
from core.limiter import limiter
import structlog
import werkzeug.utils

router = APIRouter()
logger = structlog.get_logger()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def detect_columns(df: pd.DataFrame) -> dict:
    """Heuristically map DataFrame columns to the canonical schema."""
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for canonical, candidates in {
        "date":     ["date", "day", "period", "time", "timestamp", "week"],
        "sku_id":   ["sku", "sku_id", "product", "item", "product_id", "item_id"],
        "quantity": ["quantity", "qty", "demand", "sales", "units", "amount"],
        "store_id": ["store", "store_id", "location", "site"],
        "price":    ["price", "unit_price", "cost"],
    }.items():
        for c in candidates:
            if c in cols:
                mapping[canonical] = cols[c]
                break
                
    # Detect Multi-Echelon nodes
    import re
    def extract_nodes(prefix, all_cols):
        node_names = set()
        matched_cols = []
        for c in all_cols:
            if prefix in c.lower():
                matched_cols.append(c)
                # Try to extract the node name, e.g. supplier_A_capacity -> supplier_A
                # or retailer_paris_demand -> retailer_paris
                m = re.search(f"{prefix}_([a-zA-Z0-9]+)", c.lower())
                if m:
                    node_names.add(m.group(1))
                else:
                    node_names.add(c) # fallback
        return matched_cols, list(node_names)

    retailer_cols, retailer_names = extract_nodes("retailer", cols.values())
    distributor_cols, distributor_names = extract_nodes("distributor", cols.values())
    supplier_cols, supplier_names = extract_nodes("supplier", cols.values())
    
    if retailer_cols or distributor_cols or supplier_cols:
        mapping["multi_echelon"] = {
            "retailers": retailer_cols,
            "retailer_names": retailer_names,
            "distributors": distributor_cols,
            "distributor_names": distributor_names,
            "suppliers": supplier_cols,
            "supplier_names": supplier_names
        }

    return mapping


def compute_eda(df: pd.DataFrame, col_map: dict) -> dict:
    """Compute basic EDA statistics for the dataset."""
    summary: dict = {"row_count": len(df), "column_count": len(df.columns)}

    multi = col_map.get("multi_echelon")
    if multi:
        summary["multi_echelon"] = True
        summary["num_retailers"] = len(multi.get("retailer_names", []))
        summary["num_distributors"] = len(multi.get("distributor_names", []))
        summary["num_suppliers"] = len(multi.get("supplier_names", []))
        summary["retailer_names"] = multi.get("retailer_names", [])
        summary["distributor_names"] = multi.get("distributor_names", [])
        summary["supplier_names"] = multi.get("supplier_names", [])
        
        # Total demand across retailers
        total_demand_series = pd.Series(0, index=df.index)
        for rc in multi.get("retailers", []):
            if "demand" in rc.lower() or "sales" in rc.lower():
                total_demand_series += pd.to_numeric(df[rc], errors="coerce").fillna(0)
                
        if total_demand_series.sum() > 0:
            summary["demand_mean"]   = round(float(total_demand_series.mean()), 2)
            summary["demand_std"]    = round(float(total_demand_series.std()), 2)
            summary["demand_min"]    = round(float(total_demand_series.min()), 2)
            summary["demand_max"]    = round(float(total_demand_series.max()), 2)
            summary["demand_median"] = round(float(total_demand_series.median()), 2)

    qty_col = col_map.get("quantity")
    if qty_col and qty_col in df.columns and not multi:
        series = pd.to_numeric(df[qty_col], errors="coerce").dropna()
        summary["demand_mean"]   = round(float(series.mean()), 2)
        summary["demand_std"]    = round(float(series.std()), 2)
        summary["demand_min"]    = round(float(series.min()), 2)
        summary["demand_max"]    = round(float(series.max()), 2)
        summary["demand_median"] = round(float(series.median()), 2)

    date_col = col_map.get("date")
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(dates) > 0:
            summary["date_min"] = str(dates.min().date())
            summary["date_max"] = str(dates.max().date())
            summary["num_periods"] = int((dates.max() - dates.min()).days)

    sku_col = col_map.get("sku_id")
    if sku_col and sku_col in df.columns:
        top = df[sku_col].value_counts().head(10)
        summary["top_skus"] = top.index.tolist()
        summary["num_skus"]  = int(df[sku_col].nunique())

    store_col = col_map.get("store_id")
    if store_col and store_col in df.columns:
        summary["num_stores"] = int(df[store_col].nunique())

    return summary


@router.post("/", response_model=DatasetOut, status_code=201)
@limiter.limit("5/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    column_mapping: Optional[str] = Form(None),  # JSON string override
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Validate extension
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")

    # Sanitize original filename securely
    original_name = getattr(file, "filename", "")
    try:
        from werkzeug.utils import secure_filename
        clean_name = secure_filename(original_name)
    except ImportError:
        clean_name = "".join(c for c in original_name if c.isalnum() or c in (".", "_", "-"))
        
    dataset_id = str(uuid.uuid4())
    safe_name = f"{dataset_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Parse with pandas
    try:
        if ext.lower() == ".csv":
            df = pd.read_csv(file_path, nrows=100_000)
        else:
            df = pd.read_excel(file_path, nrows=100_000)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    # Column detection
    if column_mapping:
        try:
            col_map = json.loads(column_mapping)
        except json.JSONDecodeError:
            col_map = detect_columns(df)
    else:
        col_map = detect_columns(df)

    eda = compute_eda(df, col_map)

    dataset = Dataset(
        id=dataset_id,
        owner_id=current_user.id,
        filename=safe_name,
        original_name=clean_name,
        file_path=file_path,
        file_size=len(content),
        row_count=len(df),
        columns=col_map,
        eda_summary=eda,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


@router.get("/{dataset_id}", response_model=DatasetOut)
async def get_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.owner_id == current_user.id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    rows: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.owner_id == current_user.id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    _, ext = os.path.splitext(dataset.filename)
    if ext == ".csv":
        df = pd.read_csv(dataset.file_path, nrows=rows)
    else:
        df = pd.read_excel(dataset.file_path, nrows=rows)

    return {
        "columns": list(df.columns),
        "rows": df.fillna("").to_dict(orient="records"),
    }

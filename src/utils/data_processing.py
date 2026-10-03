"""
M5 Data Processing Pipeline
=============================
Downloads, processes, and prepares the M5 Walmart dataset
for use in the supply chain simulation.

M5 Dataset Structure:
- sales_train_validation.csv: Daily sales for 30,490 products across 10 stores
- calendar.csv: Date info, events, SNAP days
- sell_prices.csv: Weekly prices per store/item

We process this into demand sequences for our multi-echelon simulation.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import json


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "m5"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class M5DataProcessor:
    """
    Processes M5 dataset for supply chain simulation.
    
    Pipeline:
    1. Load raw M5 files
    2. Select subset (configurable stores/departments)
    3. Aggregate to desired granularity
    4. Create train/val/test splits
    5. Generate demand sequences for environment
    6. Create features for forecasting
    """
    
    def __init__(
        self,
        stores: Optional[List[str]] = None,
        departments: Optional[List[str]] = None,
        aggregation_level: str = "department",  # "item", "category", "department", "store"
        num_retailers: int = 3,
    ):
        self.stores = stores or ["CA_1", "CA_2", "CA_3"]  # Default: 3 California stores
        self.departments = departments or ["FOODS_1", "FOODS_2", "FOODS_3"]
        self.aggregation_level = aggregation_level
        self.num_retailers = num_retailers
        
        # Data containers
        self.sales_df = None
        self.calendar_df = None
        self.prices_df = None
        self.demand_matrix = None
    
    def check_data_exists(self) -> bool:
        """Check if raw M5 data is available."""
        required_files = [
            RAW_DIR / "sales_train_validation.csv",
            RAW_DIR / "calendar.csv",
            RAW_DIR / "sell_prices.csv",
        ]
        return all(f.exists() for f in required_files)
    
    def download_instructions(self) -> str:
        """Return instructions for downloading M5 data."""
        return """
╔══════════════════════════════════════════════════════╗
║           M5 DATASET DOWNLOAD INSTRUCTIONS           ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Option 1: Kaggle CLI                                ║
║  > kaggle competitions download -c m5-forecasting-   ║
║    accuracy -p data/raw/m5/                          ║
║                                                      ║
║  Option 2: Manual Download                           ║
║  1. Go to: https://www.kaggle.com/competitions/      ║
║     m5-forecasting-accuracy/data                     ║
║  2. Download these files to data/raw/m5/:            ║
║     - sales_train_validation.csv                     ║
║     - calendar.csv                                   ║
║     - sell_prices.csv                                ║
║                                                      ║
║  Option 3: On Kaggle Notebook (recommended)          ║
║  The data is already available at:                   ║
║  /kaggle/input/m5-forecasting-accuracy/              ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
        """
    
    def load_raw_data(self, data_path: str = None):
        """Load raw M5 CSV files."""
        base = Path(data_path) if data_path else RAW_DIR
        
        print("Loading M5 data...")
        self.sales_df = pd.read_csv(base / "sales_train_validation.csv")
        self.calendar_df = pd.read_csv(base / "calendar.csv")
        self.prices_df = pd.read_csv(base / "sell_prices.csv")
        
        print(f"  Sales shape:    {self.sales_df.shape}")
        print(f"  Calendar shape: {self.calendar_df.shape}")
        print(f"  Prices shape:   {self.prices_df.shape}")
        
        return self
    
    def filter_subset(self):
        """Filter to selected stores and departments."""
        if self.sales_df is None:
            raise RuntimeError("Load data first with load_raw_data()")
        
        # Filter stores
        mask = self.sales_df["store_id"].isin(self.stores)
        
        # Filter departments
        if self.departments:
            mask &= self.sales_df["dept_id"].isin(self.departments)
        
        self.sales_df = self.sales_df[mask].reset_index(drop=True)
        print(f"  Filtered to {len(self.sales_df)} items across {len(self.stores)} stores")
        
        return self
    
    def aggregate_demand(self) -> pd.DataFrame:
        """
        Aggregate sales to the desired granularity.
        Returns a DataFrame with columns = dates, rows = aggregated groups.
        """
        if self.sales_df is None:
            raise RuntimeError("Load data first")
        
        # Get day columns (d_1, d_2, ..., d_1913)
        day_cols = [c for c in self.sales_df.columns if c.startswith("d_")]
        
        if self.aggregation_level == "store":
            # Aggregate all items per store
            grouped = self.sales_df.groupby("store_id")[day_cols].sum()
        elif self.aggregation_level == "department":
            # Aggregate per store-department
            grouped = self.sales_df.groupby(["store_id", "dept_id"])[day_cols].sum()
        elif self.aggregation_level == "category":
            # Aggregate per store-category
            grouped = self.sales_df.groupby(["store_id", "cat_id"])[day_cols].sum()
        else:  # item level
            grouped = self.sales_df.set_index("id")[day_cols]
        
        print(f"  Aggregated to {len(grouped)} demand series at '{self.aggregation_level}' level")
        
        return grouped
    
    def create_demand_matrix(self) -> np.ndarray:
        """
        Create demand matrix for the environment.
        Shape: (num_periods, num_retailers)
        
        Maps aggregated demand series to retailers in the simulation.
        """
        grouped = self.aggregate_demand()
        
        # Convert to numpy: shape (num_series, num_days)
        demand_full = grouped.values.astype(np.float32)
        
        # Select/combine series to match num_retailers
        if len(demand_full) >= self.num_retailers:
            # Take the first num_retailers series
            demand_selected = demand_full[:self.num_retailers]
        else:
            # Pad with copies if not enough series
            repeats = (self.num_retailers // len(demand_full)) + 1
            demand_selected = np.tile(demand_full, (repeats, 1))[:self.num_retailers]
        
        # Transpose: (num_days, num_retailers)
        self.demand_matrix = demand_selected.T
        
        print(f"  Demand matrix shape: {self.demand_matrix.shape} (periods × retailers)")
        
        return self.demand_matrix
    
    def create_features(self) -> Dict[str, pd.DataFrame]:
        """
        Create feature matrices for the forecasting model.
        
        Features per time step:
        - Lag features (1, 7, 14, 28 days)
        - Rolling statistics (7, 14, 28 day mean/std)
        - Day of week, month, year
        - Event flags (from calendar)
        - Price features
        """
        if self.demand_matrix is None:
            self.create_demand_matrix()
        
        features = {}
        
        for retailer_idx in range(self.num_retailers):
            demand = self.demand_matrix[:, retailer_idx]
            
            df = pd.DataFrame({"demand": demand})
            
            # Lag features
            for lag in [1, 7, 14, 28]:
                df[f"lag_{lag}"] = df["demand"].shift(lag)
            
            # Rolling statistics
            for window in [7, 14, 28]:
                df[f"rolling_mean_{window}"] = df["demand"].rolling(window).mean()
                df[f"rolling_std_{window}"] = df["demand"].rolling(window).std()
                df[f"rolling_min_{window}"] = df["demand"].rolling(window).min()
                df[f"rolling_max_{window}"] = df["demand"].rolling(window).max()
            
            # Calendar features
            if self.calendar_df is not None:
                num_days = min(len(demand), len(self.calendar_df))
                cal = self.calendar_df.iloc[:num_days]
                
                df["day_of_week"] = pd.to_datetime(cal["date"].values[:len(df)]).dayofweek
                df["month"] = pd.to_datetime(cal["date"].values[:len(df)]).month
                df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
                
                # Events
                if "event_name_1" in cal.columns:
                    df["has_event"] = cal["event_name_1"].notna().astype(int).values[:len(df)]
                
                # SNAP days (food stamp)
                snap_cols = [c for c in cal.columns if c.startswith("snap_")]
                for col in snap_cols[:1]:  # Take first store's SNAP
                    df["snap"] = cal[col].values[:len(df)]
            
            # Drop NaN rows (from lag/rolling features)
            df = df.fillna(0)
            
            features[f"retailer_{retailer_idx}"] = df
        
        return features
    
    def create_train_val_test_split(
        self,
        val_days: int = 28,
        test_days: int = 28,
    ) -> Dict[str, np.ndarray]:
        """
        Split demand data into train/val/test (time-based split).
        """
        if self.demand_matrix is None:
            self.create_demand_matrix()
        
        total_days = len(self.demand_matrix)
        
        test_start = total_days - test_days
        val_start = test_start - val_days
        
        splits = {
            "train": self.demand_matrix[:val_start],
            "val": self.demand_matrix[val_start:test_start],
            "test": self.demand_matrix[test_start:],
        }
        
        print(f"  Train: {splits['train'].shape[0]} days")
        print(f"  Val:   {splits['val'].shape[0]} days")
        print(f"  Test:  {splits['test'].shape[0]} days")
        
        return splits
    
    def save_processed(self, output_dir: str = None):
        """Save all processed data."""
        out = Path(output_dir) if output_dir else PROCESSED_DIR
        out.mkdir(parents=True, exist_ok=True)
        
        # Save demand matrix
        if self.demand_matrix is not None:
            np.save(out / "demand_matrix.npy", self.demand_matrix)
        
        # Save splits
        splits = self.create_train_val_test_split()
        for name, data in splits.items():
            np.save(out / f"demand_{name}.npy", data)
        
        # Save features
        features = self.create_features()
        for key, df in features.items():
            df.to_parquet(out / f"features_{key}.parquet")
        
        # Save metadata
        metadata = {
            "stores": self.stores,
            "departments": self.departments,
            "aggregation_level": self.aggregation_level,
            "num_retailers": self.num_retailers,
            "demand_shape": list(self.demand_matrix.shape),
            "num_days": int(self.demand_matrix.shape[0]),
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  ✅ All processed data saved to: {out}")
    
    @classmethod
    def load_processed(cls, data_dir: str = None) -> Dict[str, np.ndarray]:
        """Load previously processed data."""
        d = Path(data_dir) if data_dir else PROCESSED_DIR
        
        data = {
            "demand_matrix": np.load(d / "demand_matrix.npy"),
            "demand_train": np.load(d / "demand_train.npy"),
            "demand_val": np.load(d / "demand_val.npy"),
            "demand_test": np.load(d / "demand_test.npy"),
        }
        
        with open(d / "metadata.json", "r") as f:
            data["metadata"] = json.load(f)
        
        print(f"  Loaded processed data from: {d}")
        print(f"  Demand matrix: {data['demand_matrix'].shape}")
        
        return data


def generate_synthetic_demand(
    num_periods: int = 365,
    num_retailers: int = 3,
    base_demand: float = 20.0,
    seasonality: bool = True,
    trend: float = 0.01,
    noise_std: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic demand data for testing (no M5 needed).
    
    Returns: shape (num_periods, num_retailers)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_periods)
    
    demand = np.zeros((num_periods, num_retailers))
    
    for r in range(num_retailers):
        # Base level (slightly different per retailer)
        base = base_demand * (1 + 0.2 * rng.random())
        
        # Trend
        trend_component = trend * t
        
        # Seasonality (weekly + yearly)
        if seasonality:
            weekly = 3 * np.sin(2 * np.pi * t / 7)
            yearly = 5 * np.sin(2 * np.pi * t / 365)
            seasonal = weekly + yearly
        else:
            seasonal = 0
        
        # Random noise
        noise = rng.normal(0, noise_std, num_periods)
        
        # Combine and ensure non-negative
        demand[:, r] = np.maximum(0, base + trend_component + seasonal + noise)
    
    return demand.astype(np.float32)

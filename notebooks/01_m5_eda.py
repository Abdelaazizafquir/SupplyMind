"""
M5 EDA & Data Preparation Script
===================================
Run this script to explore the M5 dataset and prepare data
for the supply chain simulation.

Usage:
    python notebooks/01_m5_eda.py --data_path data/raw/m5/

For Kaggle: data is at /kaggle/input/m5-forecasting-accuracy/
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.utils.data_processing import M5DataProcessor, generate_synthetic_demand


def set_plot_style():
    """Set premium plot aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (14, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.dpi": 120,
    })
    

def eda_overview(processor: M5DataProcessor, save_dir: Path):
    """Generate overview statistics and plots."""
    print("\n" + "=" * 60)
    print("  M5 DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    sales = processor.sales_df
    calendar = processor.calendar_df
    
    # Basic stats
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    print(f"\n  Total items:        {len(sales):,}")
    print(f"  Total days:         {len(day_cols):,}")
    print(f"  Stores:             {sales['store_id'].nunique()}")
    print(f"  Departments:        {sales['dept_id'].nunique()}")
    print(f"  Categories:         {sales['cat_id'].nunique()}")
    
    # 1. Sales Distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Per store total sales
    store_sales = sales.groupby("store_id")[day_cols].sum().sum(axis=1)
    store_sales.plot(kind="bar", ax=axes[0], color="#4CAF50", edgecolor="white")
    axes[0].set_title("Total Sales by Store")
    axes[0].set_ylabel("Total Units Sold")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Per department 
    dept_sales = sales.groupby("dept_id")[day_cols].sum().sum(axis=1).sort_values(ascending=False)
    dept_sales.plot(kind="bar", ax=axes[1], color="#2196F3", edgecolor="white")
    axes[1].set_title("Total Sales by Department")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Per category
    cat_sales = sales.groupby("cat_id")[day_cols].sum().sum(axis=1)
    cat_sales.plot(kind="bar", ax=axes[2], color="#FF9800", edgecolor="white")
    axes[2].set_title("Total Sales by Category")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / "01_sales_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✅ Sales distribution plot saved")
    
    return sales, calendar


def eda_time_series(processor: M5DataProcessor, save_dir: Path):
    """Analyze time series patterns."""
    sales = processor.sales_df
    calendar = processor.calendar_df
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    
    # Aggregate daily total sales
    daily_total = sales[day_cols].sum(axis=0)
    
    # Map to dates
    dates = pd.to_datetime(calendar["date"].iloc[:len(daily_total)])
    
    # 2. Time series plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Overall trend
    axes[0].plot(dates, daily_total.values, alpha=0.5, linewidth=0.8, color="#2196F3")
    rolling_mean = pd.Series(daily_total.values).rolling(28).mean()
    axes[0].plot(dates, rolling_mean, color="#F44336", linewidth=2, label="28-day MA")
    axes[0].set_title("Daily Total Sales (All Stores)")
    axes[0].legend()
    axes[0].set_ylabel("Units")
    
    # Per-store daily sales (top 3 stores)
    stores = processor.stores[:3]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    for store, color in zip(stores, colors):
        store_data = sales[sales["store_id"] == store][day_cols].sum()
        rolling = pd.Series(store_data.values).rolling(28).mean()
        axes[1].plot(dates, rolling, label=store, linewidth=1.5, color=color)
    axes[1].set_title("Daily Sales by Store (28-day MA)")
    axes[1].legend()
    axes[1].set_ylabel("Units")
    
    # Weekly seasonality
    if len(daily_total) > 7:
        weekly_pattern = []
        for dow in range(7):
            mask = dates.dayofweek == dow
            weekly_pattern.append(daily_total.values[mask.values[:len(daily_total)]].mean())
        
        days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        axes[2].bar(days_of_week, weekly_pattern, color="#9C27B0", edgecolor="white")
        axes[2].set_title("Average Sales by Day of Week")
        axes[2].set_ylabel("Average Daily Units")
    
    plt.tight_layout()
    plt.savefig(save_dir / "02_time_series_patterns.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✅ Time series patterns plot saved")


def eda_demand_matrix(processor: M5DataProcessor, save_dir: Path):
    """Visualize the demand matrix for simulation."""
    demand = processor.create_demand_matrix()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 3a. Demand heatmap
    im = axes[0, 0].imshow(
        demand[:min(365, len(demand))].T, 
        aspect="auto", cmap="YlOrRd", interpolation="nearest"
    )
    axes[0, 0].set_title("Demand Heatmap (First Year)")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Retailer")
    plt.colorbar(im, ax=axes[0, 0], label="Demand")
    
    # 3b. Demand distribution per retailer
    for r in range(processor.num_retailers):
        axes[0, 1].hist(demand[:, r], bins=50, alpha=0.6, label=f"Retailer {r}")
    axes[0, 1].set_title("Demand Distribution by Retailer")
    axes[0, 1].set_xlabel("Daily Demand")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    
    # 3c. Demand correlation between retailers
    corr = np.corrcoef(demand.T)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 0],
                xticklabels=[f"R{i}" for i in range(processor.num_retailers)],
                yticklabels=[f"R{i}" for i in range(processor.num_retailers)])
    axes[1, 0].set_title("Demand Correlation Between Retailers")
    
    # 3d. Demand statistics
    stats_data = []
    for r in range(processor.num_retailers):
        d = demand[:, r]
        stats_data.append({
            "Retailer": f"R{r}",
            "Mean": f"{d.mean():.1f}",
            "Std": f"{d.std():.1f}",
            "Min": f"{d.min():.0f}",
            "Max": f"{d.max():.0f}",
            "CV": f"{d.std()/d.mean():.3f}",
        })
    
    axes[1, 1].axis("off")
    table = axes[1, 1].table(
        cellText=[[s[k] for k in s] for s in stats_data],
        colLabels=list(stats_data[0].keys()),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    axes[1, 1].set_title("Demand Statistics Summary", pad=20)
    
    plt.tight_layout()
    plt.savefig(save_dir / "03_demand_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✅ Demand matrix visualization saved")


def eda_events_impact(processor: M5DataProcessor, save_dir: Path):
    """Analyze impact of events on demand."""
    sales = processor.sales_df
    calendar = processor.calendar_df
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    
    if "event_name_1" not in calendar.columns:
        print("  ⚠️  No event data in calendar, skipping event analysis")
        return
    
    daily_total = sales[day_cols].sum(axis=0).values
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Event vs non-event days
    event_mask = calendar["event_name_1"].notna().values[:len(daily_total)]
    
    event_days = daily_total[event_mask]
    normal_days = daily_total[~event_mask]
    
    axes[0].boxplot([normal_days, event_days], labels=["Normal Days", "Event Days"])
    axes[0].set_title("Sales: Event vs Normal Days")
    axes[0].set_ylabel("Daily Total Sales")
    
    # Top events by sales impact
    event_sales = []
    for d_idx in range(min(len(daily_total), len(calendar))):
        if pd.notna(calendar.iloc[d_idx].get("event_name_1")):
            event_sales.append({
                "event": calendar.iloc[d_idx]["event_name_1"],
                "sales": daily_total[d_idx],
            })
    
    if event_sales:
        event_df = pd.DataFrame(event_sales)
        top_events = event_df.groupby("event")["sales"].mean().sort_values(ascending=False).head(10)
        top_events.plot(kind="barh", ax=axes[1], color="#E91E63", edgecolor="white")
        axes[1].set_title("Top Events by Average Sales")
        axes[1].set_xlabel("Average Daily Sales")
    
    plt.tight_layout()
    plt.savefig(save_dir / "04_events_impact.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✅ Event impact analysis saved")


def run_synthetic_demo(save_dir: Path):
    """Run EDA on synthetic data when M5 is not available."""
    print("\n  ⚠️  M5 data not found. Running with SYNTHETIC data for demo.")
    print("  Download M5 data for real experiments.\n")
    
    demand = generate_synthetic_demand(num_periods=365, num_retailers=3)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Time series
    for r in range(3):
        axes[0, 0].plot(demand[:, r], label=f"Retailer {r}", alpha=0.8)
    axes[0, 0].set_title("Synthetic Demand - Daily")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Demand")
    
    # Distribution
    for r in range(3):
        axes[0, 1].hist(demand[:, r], bins=30, alpha=0.5, label=f"Retailer {r}")
    axes[0, 1].set_title("Demand Distribution")
    axes[0, 1].legend()
    
    # Heatmap
    im = axes[1, 0].imshow(demand.T, aspect="auto", cmap="YlOrRd")
    axes[1, 0].set_title("Demand Heatmap")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Retailer")
    plt.colorbar(im, ax=axes[1, 0])
    
    # Autocorrelation
    for r in range(3):
        acf = np.correlate(demand[:, r] - demand[:, r].mean(), 
                          demand[:, r] - demand[:, r].mean(), mode="full")
        acf = acf[len(acf)//2:]
        acf /= acf[0]
        axes[1, 1].plot(acf[:50], label=f"Retailer {r}", alpha=0.8)
    axes[1, 1].set_title("Autocorrelation (first 50 lags)")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("Lag")
    
    plt.tight_layout()
    plt.savefig(save_dir / "00_synthetic_demand_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # Save synthetic data
    np.save(PROCESSED_DIR / "demand_matrix.npy", demand)
    
    # Create splits
    train = demand[:300]
    val = demand[300:335]
    test = demand[335:]
    np.save(PROCESSED_DIR / "demand_train.npy", train)
    np.save(PROCESSED_DIR / "demand_val.npy", val)
    np.save(PROCESSED_DIR / "demand_test.npy", test)
    
    print(f"\n  ✅ Synthetic data saved to {PROCESSED_DIR}")
    print(f"     Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")


# Output directory
PROCESSED_DIR = project_root / "data" / "processed"
PLOTS_DIR = project_root / "experiments" / "eda_plots"


def main():
    parser = argparse.ArgumentParser(description="M5 Dataset EDA")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to M5 raw data directory")
    parser.add_argument("--stores", nargs="+", default=["CA_1", "CA_2", "CA_3"])
    parser.add_argument("--departments", nargs="+", default=["FOODS_1", "FOODS_2", "FOODS_3"])
    parser.add_argument("--aggregation", type=str, default="department",
                       choices=["item", "category", "department", "store"])
    parser.add_argument("--num_retailers", type=int, default=3)
    args = parser.parse_args()
    
    # Create directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    set_plot_style()
    
    # Initialize processor
    processor = M5DataProcessor(
        stores=args.stores,
        departments=args.departments,
        aggregation_level=args.aggregation,
        num_retailers=args.num_retailers,
    )
    
    # Check if M5 data exists
    data_path = args.data_path or str(project_root / "data" / "raw" / "m5")
    
    if processor.check_data_exists() or (args.data_path and Path(args.data_path).exists()):
        # Full M5 EDA
        processor.load_raw_data(data_path)
        processor.filter_subset()
        
        eda_overview(processor, PLOTS_DIR)
        eda_time_series(processor, PLOTS_DIR)
        eda_demand_matrix(processor, PLOTS_DIR)
        eda_events_impact(processor, PLOTS_DIR)
        
        # Save processed data
        processor.save_processed()
        
        print("\n" + "=" * 60)
        print("  ✅ EDA COMPLETE - All plots and data saved!")
        print("=" * 60)
    else:
        # Run with synthetic data
        print(processor.download_instructions())
        run_synthetic_demo(PLOTS_DIR)


if __name__ == "__main__":
    main()

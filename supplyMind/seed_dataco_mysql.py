import os
import sys
import pandas as pd
import pymysql
from datetime import datetime
import numpy as np

# Fix for printing emojis in Windows terminals
sys.stdout.reconfigure(encoding='utf-8')

# --- Configuration ---
# 1. Download "DataCoSupplyChainDataset.csv" from Kaggle:
#    https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
# 2. Place it in the same folder as this script.
DATACO_CSV_PATH = "DataCoSupplyChainDataset.csv"

# MySQL Connection Configuration
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""  # Update with your password
DB_NAME = "supply_chain_db"
TABLE_NAME = "inventory_dataco"

def process_and_seed():
    if not os.path.exists(DATACO_CSV_PATH):
        print(f"❌ Error: Could not find {DATACO_CSV_PATH}")
        print("Please download it from Kaggle and place it in this directory.")
        return

    print(f"📥 Loading DataCo dataset from {DATACO_CSV_PATH} (this might take a moment)...")
    try:
        # We only need specific columns to save memory
        cols_to_use = ['Order City', 'Category Name', 'Order Region', 'Order Item Quantity']
        df = pd.read_csv(DATACO_CSV_PATH, usecols=cols_to_use, encoding='latin1')
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print("⚙️ Processing transactional data into supply chain nodes...")
    nodes = []

    # 1. RETAILERS (Based on Order Cities)
    # Group by city to find total demand
    city_demand = df.groupby('Order City')['Order Item Quantity'].sum()
    
    # Scale down the demand to make it a realistic "daily" snapshot rather than total historical sum
    # Let's say the dataset spans ~3 years (approx 1000 days)
    days_in_dataset = 1000 
    
    for city, total_qty in city_demand.items():
        daily_demand = max(5, int(total_qty / days_in_dataset))
        capacity = daily_demand * 15  # 15 days of capacity
        inventory = int(capacity * np.random.uniform(0.1, 0.8)) # random current stock
        
        # Ensure name isn't too long
        safe_name = str(city)[:50]
        
        nodes.append({
            "node_id": f"RET_{len(nodes)}",
            "node_name": f"City: {safe_name}",
            "node_type": "retailer",
            "capacity": capacity,
            "inventory": inventory,
            "demand": daily_demand
        })

    # 2. SUPPLIERS (Based on Category Names)
    dept_demand = df.groupby('Category Name')['Order Item Quantity'].sum()
    for dept, total_qty in dept_demand.items():
        daily_demand = max(50, int(total_qty / days_in_dataset))
        capacity = daily_demand * 30  # Suppliers hold 30 days of capacity
        inventory = int(capacity * np.random.uniform(0.3, 0.9))
        
        nodes.append({
            "node_id": f"SUP_{len(nodes)}",
            "node_name": f"Category: {str(dept)[:50]}",
            "node_type": "supplier",
            "capacity": capacity,
            "inventory": inventory,
            "demand": daily_demand
        })

    # 3. DISTRIBUTORS (Connecting Suppliers to Cities based on Regions)
    region_demand = df.groupby('Order Region')['Order Item Quantity'].sum()
    
    for region, total_qty in region_demand.items():
        daily_demand = max(20, int(total_qty / days_in_dataset))
        capacity = daily_demand * 20
        inventory = int(capacity * np.random.uniform(0.2, 0.8))
        
        nodes.append({
            "node_id": f"DIST_{len(nodes)}",
            "node_name": f"Hub: {str(region)[:50]}",
            "node_type": "distributor",
            "capacity": capacity,
            "inventory": inventory,
            "demand": daily_demand
        })

    print(f"✅ Generated {len(nodes)} distinct supply chain nodes from DataCo data.")

    # Push to MySQL
    print(f"🔌 Connecting to MySQL server at {DB_HOST}...")
    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, autocommit=True)
        cursor = conn.cursor()
        
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        cursor.execute(f"USE {DB_NAME}")
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                node_id VARCHAR(50) NOT NULL UNIQUE,
                node_name VARCHAR(100) NOT NULL,
                node_type VARCHAR(50) NOT NULL,
                capacity INT NOT NULL,
                inventory INT NOT NULL,
                demand INT NOT NULL,
                last_updated DATETIME
            )
        """)
        
        cursor.execute(f"TRUNCATE TABLE {TABLE_NAME}")
        
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for node in nodes:
            cursor.execute(f"""
                INSERT INTO {TABLE_NAME} (node_id, node_name, node_type, capacity, inventory, demand, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (node["node_id"], node["node_name"], node["node_type"], node["capacity"], node["inventory"], node["demand"], now))
            
        print(f"🎉 Success! Inserted {len(nodes)} rows into {DB_NAME}.{TABLE_NAME}")
        
    except Exception as e:
        print(f"❌ MySQL Error: {e}")
    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

if __name__ == "__main__":
    process_and_seed()

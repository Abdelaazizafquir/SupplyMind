import pymysql
import random
from datetime import datetime

# MySQL Connection Configuration
# Please update these if your local MySQL uses different credentials
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""  # Update with your MySQL password if you have one
DB_NAME = "supply_chain_db"

def seed_database():
    print(f"Connecting to MySQL server at {DB_HOST}...")
    try:
        # Connect without selecting a database first to create it if it doesn't exist
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            autocommit=True
        )
        cursor = conn.cursor()
        
        print(f"Creating database '{DB_NAME}' if it doesn't exist...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        cursor.execute(f"USE {DB_NAME}")
        
        table_name = "inventory_live"
        print(f"Creating table '{table_name}'...")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
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
        
        # Clear existing data
        cursor.execute(f"TRUNCATE TABLE {table_name}")
        
        # Realistic supply chain data
        # Retailers face end-consumer demand.
        # Distributors face demand from retailers.
        # Suppliers face demand from distributors.
        nodes = [
            # Retailers (Smaller capacity, lower inventory, moderate demand)
            {"id": "RET_NY_001", "name": "New York Store", "type": "retailer", "cap": 1200, "inv": random.randint(100, 400), "dem": random.randint(20, 50)},
            {"id": "RET_LA_002", "name": "Los Angeles Store", "type": "retailer", "cap": 1500, "inv": random.randint(150, 500), "dem": random.randint(30, 60)},
            {"id": "RET_CHI_003", "name": "Chicago Store", "type": "retailer", "cap": 1000, "inv": random.randint(50, 250), "dem": random.randint(15, 35)},
            {"id": "RET_MIA_004", "name": "Miami Store", "type": "retailer", "cap": 800, "inv": random.randint(80, 200), "dem": random.randint(10, 25)},
            {"id": "RET_TX_005", "name": "Dallas Store", "type": "retailer", "cap": 1100, "inv": random.randint(200, 600), "dem": random.randint(25, 45)},
            
            # Distributors (Medium capacity, hold inventory for retailers)
            {"id": "DIST_EAST_01", "name": "East Coast Hub", "type": "distributor", "cap": 5000, "inv": random.randint(1000, 2500), "dem": random.randint(100, 200)},
            {"id": "DIST_WEST_02", "name": "West Coast Hub", "type": "distributor", "cap": 6000, "inv": random.randint(1200, 3000), "dem": random.randint(150, 250)},
            {"id": "DIST_CENTRAL_03", "name": "Central Hub", "type": "distributor", "cap": 4500, "inv": random.randint(800, 2000), "dem": random.randint(80, 150)},
            
            # Suppliers (Huge capacity, source of raw materials/products)
            {"id": "SUPP_GLOBAL_01", "name": "Global Manufacturer A", "type": "supplier", "cap": 20000, "inv": random.randint(8000, 15000), "dem": random.randint(400, 800)},
            {"id": "SUPP_LOCAL_02", "name": "Domestic Supplier B", "type": "supplier", "cap": 10000, "inv": random.randint(3000, 7000), "dem": random.randint(200, 500)},
        ]
        
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print("Inserting realistic supply chain data...")
        for node in nodes:
            # Randomly simulate some critical/low inventory scenarios
            if random.random() < 0.2:  # 20% chance of being in critical stock out
                node["inv"] = int(node["cap"] * random.uniform(0.01, 0.08))
                
            cursor.execute(f"""
                INSERT INTO {table_name} (node_id, node_name, node_type, capacity, inventory, demand, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (node["id"], node["name"], node["type"], node["cap"], node["inv"], node["dem"], now))
            
            print(f"  -> Inserted {node['name']} ({node['type']}) - Stock: {node['inv']}/{node['cap']}")
            
        print("\n✅ Success! Database seeded.")
        print(f"You can now connect to:")
        print(f"  Host: {DB_HOST}")
        print(f"  Database: {DB_NAME}")
        print(f"  Table: {table_name}")
        
    except ImportError:
        print("❌ Error: pymysql is not installed. Please run: pip install pymysql")
    except Exception as e:
        print(f"❌ Error connecting to MySQL: {e}")
        print("Please ensure your MySQL server is running and the credentials in this script are correct.")
    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

if __name__ == "__main__":
    seed_database()

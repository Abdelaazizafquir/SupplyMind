import sqlite3

conn = sqlite3.connect("supplymind.db")
cursor = conn.cursor()

# Check if column already exists
cursor.execute("PRAGMA table_info(projects)")
cols = [row[1] for row in cursor.fetchall()]
print("Existing columns:", cols)

if "mode" not in cols:
    cursor.execute("ALTER TABLE projects ADD COLUMN mode TEXT DEFAULT 'simulation'")
    conn.commit()
    print("Added 'mode' column to projects table.")
else:
    print("Column 'mode' already exists.")

# Also fix existing projects that have mode stored in topology JSON
cursor.execute("SELECT id, topology FROM projects WHERE mode IS NULL OR mode = 'simulation'")
rows = cursor.fetchall()
import json
for row_id, topo_json in rows:
    if topo_json:
        try:
            topo = json.loads(topo_json) if isinstance(topo_json, str) else topo_json
            if isinstance(topo, dict) and "mode" in topo:
                cursor.execute("UPDATE projects SET mode = ? WHERE id = ?", (topo["mode"], row_id))
                print(f"Migrated project {row_id}: mode={topo['mode']}")
        except Exception as e:
            print(f"Error migrating {row_id}: {e}")

conn.commit()
conn.close()
print("Migration complete.")

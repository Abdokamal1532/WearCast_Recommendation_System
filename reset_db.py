import pyodbc
import sys

# Configuration
DB_CONNECTION_STRING = "Driver={ODBC Driver 17 for SQL Server};Server=localhost\\MSSQLSERVER01;Database=WearCastDb;Trusted_Connection=yes;"

def get_db_connection():
    return pyodbc.connect(DB_CONNECTION_STRING)

def reset_database():
    print("[ResetDB] Starting database reset...")
    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    # Disable all constraints
    print("[ResetDB] Disabling all constraints...")
    cursor.execute("EXEC sp_MSforeachtable 'ALTER TABLE ? NOCHECK CONSTRAINT ALL'")

    print("[ResetDB] Wiping all data from all tables...")
    
    # Get all tables
    cursor.execute("SELECT name FROM sys.tables WHERE name NOT IN ('__EFMigrationsHistory', 'sysdiagrams')")
    tables = [row[0] for row in cursor.fetchall()]
    
    for table in tables:
        try:
            print(f"  -> Cleaning {table}...")
            cursor.execute(f"DELETE FROM {table}")
            
            # Reset Identity seed if it has one
            cursor.execute(f"SELECT OBJECTPROPERTY(OBJECT_ID('{table}'), 'TableHasIdentity')")
            if cursor.fetchone()[0] == 1:
                cursor.execute(f"DBCC CHECKIDENT ('{table}', RESEED, 0)")
        except Exception as e:
            print(f"  [Error] Failed to clean {table}: {e}")

    # Re-enable all constraints
    print("[ResetDB] Re-enabling all constraints...")
    cursor.execute("EXEC sp_MSforeachtable 'ALTER TABLE ? WITH CHECK CHECK CONSTRAINT ALL'")

    print("[ResetDB] Database reset complete!")
    conn.close()

if __name__ == "__main__":
    confirm = input("CAUTION: This will delete ALL data in WearCastDb. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        try:
            reset_database()
        except Exception as e:
            print(f"[Error] {e}")
    else:
        print("Reset aborted.")

import pyodbc
import random
import json
from datetime import datetime, timedelta
import sys
import os

# Configuration
DB_CONNECTION_STRING = "Driver={ODBC Driver 17 for SQL Server};Server=localhost\\MSSQLSERVER01;Database=WearCastDb;Trusted_Connection=yes;"

def get_db_connection():
    return pyodbc.connect(DB_CONNECTION_STRING)

def seed():
    conn = get_db_connection()
    cursor = conn.cursor()
    print("[Seeder] Connected to database.")

    # 1. Get a valid User ID
    cursor.execute("SELECT TOP 1 Id FROM AspNetUsers")
    user_row = cursor.fetchone()
    if not user_row:
        print("[Error] No users found in AspNetUsers. Please create at least one user first.")
        return
    admin_id = user_row[0]
    print(f"[Seeder] Using User ID: {admin_id} as owner.")

    # 2. Ensure Categories exist
    categories = [
        "Casual Wear",
        "Formal Attire",
        "Sportswear",
        "Winter Collection",
        "Accessories"
    ]
    
    cat_ids = []
    for name in categories:
        cursor.execute("SELECT Id FROM Categories WHERE Name = ?", name)
        row = cursor.fetchone()
        if row:
            cat_ids.append(row[0])
        else:
            cursor.execute("""
                INSERT INTO Categories (Name, CreatedOn, IsDeleted, CreatedById, IsActive, ImageUrl) 
                OUTPUT INSERTED.Id 
                VALUES (?, GETDATE(), 0, ?, 1, 'https://placeholder.com/cat.png')
            """, name, admin_id)
            cat_ids.append(cursor.fetchone()[0])
    
    print(f"[Seeder] Categories ready: {len(cat_ids)} found/created.")

    # 3. Ensure Factory exists
    # Schema: Id, Name, Email, PhoneNumber, CommercialRegisterNumber, TaxIdNumber, Description, LogoUrl, State, City, Street, BuildingNumber, IsDeleted
    cursor.execute("SELECT TOP 1 Id FROM Factories")
    f_row = cursor.fetchone()
    if f_row:
        factory_id = f_row[0]
    else:
        cursor.execute("""
            INSERT INTO Factories (Name, Email, PhoneNumber, CommercialRegisterNumber, TaxIdNumber, Description, LogoUrl, State, City, Street, BuildingNumber, IsDeleted) 
            OUTPUT INSERTED.Id 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, "Global Manufacturing", "info@globalmfg.com", "+20123456789", "CR-12345", "TAX-67890", "High quality garments", "https://placeholder.com/logo.png", "Cairo", "Cairo", "Industrial Area", "101")
        factory_id = cursor.fetchone()[0]
    print(f"[Seeder] Factory ID: {factory_id}")

    # 4. Generate 100 Products
    styles = [1, 2, 3, 4, 5]
    audiences = [1, 2, 3, 4]
    
    print("[Seeder] Generating 100 products...")
    product_ids = []
    for i in range(1, 101):
        name = f"Premium {random.choice(['Silk', 'Cotton', 'Linen', 'Wool'])} {random.choice(['Shirt', 'Pants', 'Dress', 'Jacket'])} {i}"
        desc = f"High quality apparel for {name.lower()} Enthusiasts."
        style = random.choice(styles)
        audience = random.choice(audiences)
        price = round(random.uniform(20.0, 450.0), 2)
        cat_id = random.choice(cat_ids)
        
        cursor.execute("""
            INSERT INTO DesignedProducts 
            (Name, Description, TargetAudience, DressStyle, Price, CategoryId, FactoryId, CreatedById, CreatedOn, IsDeleted, IsActive, CanvasWidth, CanvasHeight, SalesCount) 
            OUTPUT INSERTED.Id
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, GETDATE(), 0, 1, 1024, 1024, 0)
        """, name, desc, audience, style, price, cat_id, factory_id, admin_id)
        product_ids.append(cursor.fetchone()[0])
    
    print(f"[Seeder] 100 products created.")

    # 5. Generate 200 Logs
    print("[Seeder] Generating 200 activity logs...")
    types = ["click", "purchase"]
    start_date = datetime.now() - timedelta(days=30)

    for i in range(200):
        pid = random.choice(product_ids)
        etype = random.choice(types)
        ts = (start_date + timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if etype == "click":
            payload = {
                "eventType": "click",
                "userId": admin_id,
                "timestamp": ts,
                "productDetails": {"productId": str(pid)}
            }
        else: # purchase
            payload = {
                "eventType": "purchase",
                "userId": admin_id,
                "timestamp": ts,
                "products": [{"productId": str(pid), "quantity": 1, "price": 100}]
            }
            
        cursor.execute("INSERT INTO UserActivityLogs (UserId, Payload, CreatedAt) VALUES (?, ?, ?)", 
                       admin_id, json.dumps(payload), datetime.now())

    conn.commit()
    print("[Seeder] Seeding complete! (100 products, 200 logs)")
    conn.close()

if __name__ == "__main__":
    try:
        seed()
    except Exception as e:
        print(f"[Error] {e}")

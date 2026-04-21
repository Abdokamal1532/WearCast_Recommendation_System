import pyodbc
import random
import json
import uuid
from datetime import datetime, timedelta
from faker import Faker

# =============================================================================
# CONFIGURATION
# =============================================================================
DB_CONNECTION_STRING = "Driver={ODBC Driver 17 for SQL Server};Server=localhost\\MSSQLSERVER01;Database=WearCastDb;Trusted_Connection=yes;"
fake = Faker()

# Roles
ROLE_IDS = {
    "SuperAdmin": "efc6a646-310b-4571-9b0e-bdfebe29f921",
    "FactoryManager": "03205b5f-c1bd-411b-a31d-dcaff6ca0005",
    "SellerManager": "6635cd46-2b74-4e0c-8872-1fed3faa60cd",
    "ShippingCompanyManager": "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d",
    "Driver": "d1d2d3d4-d5d6-4d5d-8d9d-0d1d2d3d4d5d",
    "Customer": "91756452-0c83-4c86-8129-88698116ee37"
}

# Enums Mapping
DRESS_STYLES = [1, 2, 3, 4, 5]
TARGET_AUDIENCES = [1, 2, 3, 4, 8]
# Sizes: 11=_2XS, 12=_XS, 13=_S, 14=_M, 15=_L, 16=_XL, 17=_2XL, 18=_3XL
SIZE_VALS = [12, 13, 14, 15, 16, 17, 18]
SIZE_NAMES = {12: "XS", 13: "S", 14: "M", 15: "L", 16: "XL", 17: "2XL", 18: "3XL"}

def get_db_connection():
    return pyodbc.connect(DB_CONNECTION_STRING)

def create_user(cursor, role_name):
    uid = str(uuid.uuid4())
    username = fake.user_name() + str(random.randint(1000, 9999))
    email = f"{username}@{fake.free_email_domain()}"
    fname = fake.first_name()
    lname = fake.last_name()
    
    cursor.execute("""
        INSERT INTO AspNetUsers (Id, UserName, NormalizedUserName, Email, NormalizedEmail, EmailConfirmed, 
        SecurityStamp, ConcurrencyStamp, PhoneNumberConfirmed, TwoFactorEnabled, LockoutEnabled, AccessFailedCount,
        FirstName, LastName, IsDeleted, IsDisabled, PhoneNumber)
        VALUES (?, ?, ?, ?, ?, 1, ?, ?, 0, 0, 1, 0, ?, ?, 0, 0, ?)
    """, uid, username, username.upper(), email, email.upper(), str(uuid.uuid4()), str(uuid.uuid4()), fname, lname, fake.numerify('###########'))
    cursor.execute("INSERT INTO AspNetUserRoles (UserId, RoleId) VALUES (?, ?)", uid, ROLE_IDS[role_name])
    return uid

def seed_real_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    print("--- STARTING CORRECTED FINAL SEEDING ---")

    # 0. Roles and Admin
    for name, rid in ROLE_IDS.items():
        cursor.execute("SELECT 1 FROM AspNetRoles WHERE Id = ?", rid)
        if not cursor.fetchone():
            cursor.execute("INSERT INTO AspNetRoles (Id, Name, NormalizedName, ConcurrencyStamp, IsDefault, IsDeleted) VALUES (?, ?, ?, ?, 0, 0)", rid, name, name.upper(), str(uuid.uuid4()))
    
    cursor.execute("SELECT TOP 1 Id FROM AspNetUsers JOIN AspNetUserRoles ON AspNetUsers.Id = AspNetUserRoles.UserId WHERE RoleId = ?", ROLE_IDS["SuperAdmin"])
    row = cursor.fetchone()
    admin_id = row[0] if row else create_user(cursor, "SuperAdmin")

    # 1. Categories
    category_names = ["T-Shirts", "Jeans", "Hoodies", "Suits", "Dresses", "Activewear", "Footwear", "Accessories"]
    cat_ids = []
    for name in category_names:
        cursor.execute("SELECT Id FROM Categories WHERE Name = ?", name)
        ex = cursor.fetchone()
        if ex: cat_ids.append(ex[0])
        else:
            cursor.execute("INSERT INTO Categories (Name, CreatedOn, IsDeleted, CreatedById, IsActive, ImageUrl) OUTPUT INSERTED.Id VALUES (?, GETDATE(), 0, ?, 1, ?)", name, admin_id, fake.image_url())
            cat_ids.append(cursor.fetchone()[0])

    # 2. Factories (5)
    factory_ids = []
    for _ in range(5):
        mgr_id = create_user(cursor, "FactoryManager")
        cursor.execute("""
            INSERT INTO Factories (Name, Email, PhoneNumber, CommercialRegisterNumber, TaxIdNumber, 
                                 Description, LogoUrl, IsDeleted, State, City, Street, BuildingNumber) 
            OUTPUT INSERTED.Id VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
        """, fake.company() + " Lab", fake.company_email(), fake.numerify('###########'), 
           fake.numerify('#########'), fake.numerify('#########'), 
           fake.catch_phrase(), fake.image_url(), 
           fake.state()[:50], fake.city()[:50], fake.street_name()[:200], str(random.randint(1, 999)))
        f_id = cursor.fetchone()[0]
        factory_ids.append(f_id)
        cursor.execute("INSERT INTO FactoryManagers (UserId, FactoryId, IsDeleted) VALUES (?, ?, 0)", mgr_id, f_id)

    # 3. Sellers (10)
    seller_ids = []
    for _ in range(10):
        sm_id = create_user(cursor, "SellerManager")
        cursor.execute("""
            INSERT INTO Sellers (Name, Email, PhoneNumber, CommercialRegisterNumber, TaxIdNumber, 
                               Description, LogoUrl, IsDeleted, State, City, Street, BuildingNumber) 
            OUTPUT INSERTED.Id VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
        """, fake.company() + " Boutique", fake.company_email(), fake.numerify('###########'), 
           fake.numerify('#########'), fake.numerify('#########'), 
           fake.bs(), fake.image_url(),
           fake.state()[:50], fake.city()[:50], fake.street_name()[:200], str(random.randint(1, 999)))
        s_id = cursor.fetchone()[0]
        seller_ids.append(s_id)
        cursor.execute("INSERT INTO SellerManagers (UserId, SellerId, IsDeleted) VALUES (?, ?, 0)", sm_id, s_id)

    # 4. Shipping & Drivers
    ship_co_ids = []
    driver_ids = []
    for _ in range(3):
        scm_id = create_user(cursor, "ShippingCompanyManager")
        cursor.execute("""
            INSERT INTO ShippingCompanies (Name, Email, PhoneNumber, Description, LogoUrl, IsDeleted, 
                                        CommercialRegisterNumber, TaxIdNumber, DeliveryFee,
                                        State, City, Street, BuildingNumber) 
            OUTPUT INSERTED.Id VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
        """, fake.company() + " Logist", fake.company_email(), fake.numerify('###########'), 
           fake.catch_phrase(), fake.image_url(), fake.numerify('#########'), fake.numerify('#########'), 25.0,
           fake.state()[:50], fake.city()[:50], fake.street_name()[:200], str(random.randint(1, 999)))
        sc_id = cursor.fetchone()[0]
        ship_co_ids.append(sc_id)
        cursor.execute("INSERT INTO ShippingCompanyManagers (UserId, ShippingCompanyId, IsDeleted) VALUES (?, ?, 0)", scm_id, sc_id)
        for _ in range(2):
            d_uid = create_user(cursor, "Driver")
            cursor.execute("""
                INSERT INTO Drivers (UserId, ShippingCompanyId, IsDeleted, VehicleType, VehiclePlateNumber, NationalId,
                                    State, City, Street, BuildingNumber, Status, ProfileImageUrl) 
                OUTPUT INSERTED.Id VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, d_uid, sc_id, random.choice([1, 2, 3, 4]), fake.bothify(text='??-####'), fake.numerify('##############'),
               fake.state()[:50], fake.city()[:50], fake.street_name()[:200], str(random.randint(1, 999)), fake.image_url())
            driver_ids.append(cursor.fetchone()[0])

    # 5. Customers (50)
    customer_uids = []
    customer_ids = []
    for _ in range(50):
        c_uid = create_user(cursor, "Customer")
        cursor.execute("""
            INSERT INTO Customers (UserId, IsDeleted, ProfileImageUrl, State, City, Street, BuildingNumber) 
            OUTPUT INSERTED.Id VALUES (?, 0, ?, ?, ?, ?, ?)
        """, c_uid, fake.image_url(), fake.state()[:50], fake.city()[:50], fake.street_name()[:200], str(random.randint(1, 999)))
        customer_ids.append(cursor.fetchone()[0])
        customer_uids.append(c_uid)

    # 6. Fixed Products (100)
    fixed_product_ids = []
    fcid_map = {} # pid -> list of (cid, cname, img)
    print("[Seeder] Seeding Fixed Products...")
    for _ in range(100):
        sid = random.choice(seller_ids)
        cat_id = random.choice(cat_ids)
        # SizeDetails as JSON
        sd_json = json.dumps([{"Size": SIZE_NAMES[s], "A": round(random.uniform(30, 50), 2), "B": round(random.uniform(50, 80), 2), "C": round(random.uniform(10, 20), 2)} for s in random.sample(SIZE_VALS, 3)])
        cursor.execute("""
            INSERT INTO FixedProducts (Name, Description, TargetAudience, DressStyle, Price, CategoryId, SellerId, CreatedById, CreatedOn, IsDeleted, IsActive, SizeDetails) 
            OUTPUT INSERTED.Id VALUES (?, ?, ?, 'Casual', ?, ?, ?, ?, GETDATE(), 0, 1, ?)
        """, fake.word().capitalize() + " Wear", fake.sentence(), random.choice(TARGET_AUDIENCES), round(random.uniform(50, 500), 2), cat_id, sid, admin_id, sd_json)
        pid = cursor.fetchone()[0]
        fixed_product_ids.append(pid)
        fcid_map[pid] = []
        for _ in range(random.randint(1, 2)):
            cname = fake.color_name()
            img = fake.image_url()
            # Sizes as JSON in FixedProductColors
            sz_json = json.dumps([{"Size": SIZE_NAMES[s], "Quantity": random.randint(1, 100)} for s in random.sample(SIZE_VALS, 5)])
            cursor.execute("""
                INSERT INTO FixedProductColors (ColorName, ColorCode, ImageUrl, ProductId, CreatedById, CreatedOn, IsDeleted, IsActive, Sizes)
                OUTPUT INSERTED.Id VALUES (?, ?, ?, ?, ?, GETDATE(), 0, 1, ?)
            """, cname, fake.hex_color(), img, pid, admin_id, sz_json)
            fcid_map[pid].append((cursor.fetchone()[0], cname, img))

    # 7. Designed Products (200)
    designed_product_ids = []
    print("[Seeder] Seeding Designed Products...")
    for i in range(200):
        fid = random.choice(factory_ids)
        cat_id = random.choice(cat_ids)
        cursor.execute("""
            INSERT INTO DesignedProducts (Name, Description, TargetAudience, DressStyle, Price, CategoryId, FactoryId, CreatedById, CreatedOn, IsDeleted, IsActive, CanvasWidth, CanvasHeight, SalesCount) 
            OUTPUT INSERTED.Id VALUES (?, ?, ?, 1, ?, ?, ?, ?, GETDATE(), 0, 1, 1024, 1024, ?)
        """, f"Style {i}", fake.sentence(), random.choice(TARGET_AUDIENCES), round(random.uniform(100, 1000), 2), cat_id, fid, admin_id, random.randint(0, 100))
        pid = cursor.fetchone()[0]
        designed_product_ids.append(pid)
        cursor.execute("INSERT INTO DesignedProductColors (Name, HexCode, MainImageUrl, DesignedProductId, IsDeleted, IsActive, CreatedById, CreatedOn) VALUES (?, ?, ?, ?, 0, 1, ?, GETDATE())", fake.color_name(), fake.hex_color(), fake.image_url(), pid, admin_id)
        for sz in random.sample(SIZE_VALS, 4):
            cursor.execute("INSERT INTO DesignedProductSizeDetails (Size, A, B, C, DesignedProductId, IsDeleted) VALUES (?, 10, 20, 30, ?, 0)", sz, pid)

    # 9. Orders & Logs
    print("[Seeder] Seeding Orders and Logs...")
    for _ in range(300):
        cid = random.choice(customer_ids)
        sid = random.choice(seller_ids)
        cursor.execute("""
            INSERT INTO Orders (CustomerId, SellerId, TotalAmount, Status, RecipientName, RecipientPhoneNumber, CreatedById, CreatedOn, IsDeleted, IsActive, 
                             ShippingAddress_State, ShippingAddress_City, ShippingAddress_Street, ShippingAddress_BuildingNumber,
                             PickUpAddress_State, PickUpAddress_City, PickUpAddress_Street, PickUpAddress_BuildingNumber) 
            OUTPUT INSERTED.Id VALUES (?, ?, ?, 'Pending', ?, ?, ?, GETDATE(), 0, 1, ?, ?, ?, ?, ?, ?, ?, ?)
        """, cid, sid, 500.0, fake.name(), fake.numerify('###########'), admin_id, fake.state()[:50], fake.city()[:50], fake.street_name()[:200], "1", fake.state()[:50], fake.city()[:50], fake.street_name()[:200], "1")
        oid = cursor.fetchone()[0]
        fpid = random.choice(fixed_product_ids)
        fcid, cname, img = random.choice(fcid_map[fpid])
        cursor.execute("INSERT INTO FixedProductOrderItems (OrderId, FixedColorId, ProductName, ColorName, ImageUrl, SizeName, Quantity, UnitPrice, CreatedById, CreatedOn, IsActive, IsDeleted) VALUES (?, ?, ?, ?, ?, 'M', 1, 100, ?, GETDATE(), 1, 0)", oid, fcid, "FP", cname, img, admin_id)

    # Activity Logs
    trending_pids = random.sample(designed_product_ids, 20)
    for _ in range(5000):
        pid = random.choice(trending_pids) if random.random() < 0.7 else random.choice(designed_product_ids)
        uid = random.choice(customer_uids)
        ts = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = {"eventType": "click", "userId": uid, "timestamp": ts, "productDetails": {"productId": str(pid)}}
        cursor.execute("INSERT INTO UserActivityLogs (UserId, Payload, CreatedAt) VALUES (?, ?, ?)", uid, json.dumps(payload), datetime.now())

    conn.commit()
    print("\n[SUCCESS] Seeding complete!")
    conn.close()

if __name__ == "__main__":
    try: seed_real_data()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()

import os
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# =========================================================
# 1. Load environment variables
# =========================================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# =========================================================
# 2. Validate database configuration
# =========================================================

required_variables = {
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "DB_NAME": DB_NAME,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
}

missing_variables = [
    name for name, value in required_variables.items()
    if not value
]

if missing_variables:
    raise ValueError(
        f"Missing environment variables: {', '.join(missing_variables)}"
    )


# =========================================================
# 3. Database connection
# =========================================================

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


# =========================================================
# 4. CSV path
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "industrial_sensor_data.csv"
)


# =========================================================
# 5. Read CSV
# =========================================================

print("Reading CSV file...")

df = pd.read_csv(CSV_PATH)

print(f"CSV loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# =========================================================
# 6. Clean column names
# =========================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# =========================================================
# 7. Make sure required columns exist
# =========================================================

required_columns = [
    "well_id",
    "equipment",
    "timestamp",
    "pressure_bar",
    "temperature_c",
    "vibration_mm_s",
    "flow_rate_bbl_day",
    "production_rate_bbl_day",
    "pump_speed_rpm",
    "failure",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


# If equipment is missing, create it
if "equipment" not in df.columns:

    if "equipment_id" in df.columns:

        df["equipment"] = df["equipment_id"]

    elif "pump_id" in df.columns:

        df["equipment"] = df["pump_id"]

    else:

        df["equipment"] = "PUMP-" + df["well_id"].astype(str)


# If well_id is missing, create it
if "well_id" not in df.columns:

    if "well" in df.columns:

        df["well_id"] = df["well"]

    else:

        raise ValueError(
            "well_id column is missing from the dataset."
        )


# If failure is missing, create it
if "failure" not in df.columns:

    if "failure_flag" in df.columns:

        df["failure"] = df["failure_flag"]

    elif "target" in df.columns:

        df["failure"] = df["target"]

    else:

        raise ValueError(
            "failure column is missing from the dataset."
        )


# Re-check required columns
missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =========================================================
# 8. Select only required columns
# =========================================================

df = df[required_columns].copy()


# =========================================================
# 9. Data type conversion
# =========================================================

print("Cleaning data...")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

numeric_columns = [
    "pressure_bar",
    "temperature_c",
    "vibration_mm_s",
    "flow_rate_bbl_day",
    "production_rate_bbl_day",
    "pump_speed_rpm",
    "failure",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# =========================================================
# 10. Remove invalid rows
# =========================================================

before_cleaning = len(df)

df = df.dropna()

after_cleaning = len(df)

print(
    f"Removed invalid rows: "
    f"{before_cleaning - after_cleaning}"
)

print(
    f"Rows after cleaning: {after_cleaning}"
)


# =========================================================
# 11. Convert data types
# =========================================================

df["well_id"] = df["well_id"].astype(str)
df["equipment"] = df["equipment"].astype(str)
df["failure"] = df["failure"].astype(int)


# =========================================================
# 12. Connect to PostgreSQL
# =========================================================

print("Connecting to PostgreSQL...")

try:

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT version();")
        )

        version = result.fetchone()[0]

        print("PostgreSQL connection successful!")
        print(version)

except Exception as e:

    print("Database connection failed.")
    print(e)

    raise


# =========================================================
# 13. Load data into PostgreSQL
# =========================================================

print("Loading data into PostgreSQL...")

df.to_sql(
    "industrial_sensor_data",
    engine,
    if_exists="replace",
    index=False,
)


# =========================================================
# 14. Verify inserted data
# =========================================================

with engine.connect() as connection:

    result = connection.execute(
        text(
            """
            SELECT COUNT(*)
            FROM industrial_sensor_data;
            """
        )
    )

    count = result.scalar()


# =========================================================
# 15. Final result
# =========================================================

print()
print("=" * 60)
print("ETL PIPELINE COMPLETED")
print("=" * 60)

print(f"Records inserted: {count}")
print("Table: industrial_sensor_data")
print("Database: oil_gas_ai")

print("=" * 60)
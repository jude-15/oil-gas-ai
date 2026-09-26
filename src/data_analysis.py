import pandas as pd

# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("data/raw/industrial_sensor_data.csv")


# ==========================================
# 2. Dataset shape
# ==========================================

print("===== DATASET SHAPE =====")
print(df.shape)


# ==========================================
# 3. Columns
# ==========================================

print("\n===== COLUMNS =====")
print(df.columns.tolist())


# ==========================================
# 4. First 5 rows
# ==========================================

print("\n===== FIRST 5 ROWS =====")
print(df.head())


# ==========================================
# 5. Data types
# ==========================================

print("\n===== DATA TYPES =====")
print(df.dtypes)


# ==========================================
# 6. Missing values
# ==========================================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


# ==========================================
# 7. Duplicates
# ==========================================

print("\n===== DUPLICATES =====")
print(df.duplicated().sum())


# ==========================================
# 8. Statistical summary
# ==========================================

print("\n===== STATISTICS =====")
print(df.describe())


# ==========================================
# 9. Failure distribution
# ==========================================

print("\n===== FAILURE DISTRIBUTION =====")
print(df["failure"].value_counts())


# ==========================================
# 10. Failure percentage
# ==========================================

print("\n===== FAILURE PERCENTAGE =====")
print(df["failure"].value_counts(normalize=True) * 100)


# ==========================================
# 11. Normal vs Failure
# ==========================================

print("\n===== NORMAL VS FAILURE =====")

comparison = df.groupby("failure")[
    [
        "pressure_bar",
        "temperature_c",
        "vibration_mm_s",
        "flow_rate_bbl_day",
        "production_rate_bbl_day",
        "pump_speed_rpm"
    ]
].mean()

print(comparison.to_string())
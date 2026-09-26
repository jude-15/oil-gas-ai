import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# =========================
# PROJECT CONFIGURATION
# =========================

NUM_WELLS = 10
READINGS_PER_WELL = 500

# =========================
# GENERATE DATA
# =========================

data = []

for well_number in range(1, NUM_WELLS + 1):

    well_id = f"WELL-{well_number:03d}"
    equipment_id = f"PUMP-{well_number:03d}"

    for reading in range(READINGS_PER_WELL):

        timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(
            hours=reading
        )

        # Normal industrial sensor values
        pressure = np.random.normal(150, 8)
        temperature = np.random.normal(75, 5)
        vibration = np.random.normal(3.5, 0.6)
        flow_rate = np.random.normal(1250, 80)
        production_rate = np.random.normal(1180, 70)
        pump_speed = np.random.normal(1450, 50)

        failure = 0

        # =========================
        # SIMULATE ANOMALIES
        # =========================

        if np.random.random() < 0.05:

            pressure += np.random.uniform(15, 30)
            temperature += np.random.uniform(10, 25)
            vibration += np.random.uniform(3, 7)

            flow_rate -= np.random.uniform(100, 250)
            production_rate -= np.random.uniform(100, 250)

            # Some anomalies become failures
            if vibration > 8 and temperature > 90:
                failure = 1

        data.append([
            timestamp,
            well_id,
            equipment_id,
            pressure,
            temperature,
            vibration,
            flow_rate,
            production_rate,
            pump_speed,
            failure
        ])

# =========================
# CREATE DATAFRAME
# =========================

columns = [
    "timestamp",
    "well_id",
    "equipment_id",
    "pressure_bar",
    "temperature_c",
    "vibration_mm_s",
    "flow_rate_bbl_day",
    "production_rate_bbl_day",
    "pump_speed_rpm",
    "failure"
]

df = pd.DataFrame(data, columns=columns)

# =========================
# SAVE DATASET
# =========================

output_path = "data/raw/industrial_sensor_data.csv"

df.to_csv(output_path, index=False)

print("Dataset generated successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {output_path}")

print("\nFirst 5 rows:")
print(df.head())

print("\nFailure distribution:")
print(df["failure"].value_counts())
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("data/raw/industrial_sensor_data.csv")


# ==========================================
# 2. Create Industrial AI scatter plot
# ==========================================

normal = df[df["failure"] == 0]
failure = df[df["failure"] == 1]

plt.figure(figsize=(10, 6))

plt.scatter(
    normal["temperature_c"],
    normal["vibration_mm_s"],
    alpha=0.5,
    label="Normal"
)

plt.scatter(
    failure["temperature_c"],
    failure["vibration_mm_s"],
    alpha=0.8,
    label="Failure"
)


# ==========================================
# 3. Dashboard-style labels
# ==========================================

plt.title("Industrial Equipment Health Monitoring")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# ==========================================
# 4. Save graph
# ==========================================

plt.savefig(
    "data/processed/equipment_health_monitoring.png",
    dpi=300
)

plt.show()
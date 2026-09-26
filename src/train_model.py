import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv(
    "data/raw/industrial_sensor_data.csv"
)


# ==========================================
# 2. Select features
# ==========================================

features = [
    "pressure_bar",
    "temperature_c",
    "vibration_mm_s",
    "flow_rate_bbl_day",
    "production_rate_bbl_day",
    "pump_speed_rpm"
]

X = df[features]
y = df["failure"]


# ==========================================
# 3. Split data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("===== DATA SPLIT =====")

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ==========================================
# 4. Scale features
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
# 5. Create model
# ==========================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)


# ==========================================
# 6. Train model
# ==========================================

model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 7. Predictions
# ==========================================

y_pred = model.predict(
    X_test_scaled
)


y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# ==========================================
# 8. Model evaluation
# ==========================================

print("\n===== MODEL PERFORMANCE =====")

print(
    f"Accuracy: "
    f"{accuracy_score(y_test, y_pred):.4f}"
)


print("\n===== CONFUSION MATRIX =====")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Failure"
        ]
    )
)


# ==========================================
# 9. Risk function
# ==========================================

def get_risk_level(probability):

    if probability >= 0.80:
        return "HIGH"

    elif probability >= 0.40:
        return "MEDIUM"

    else:
        return "LOW"


# ==========================================
# 10. Risk assessment
# ==========================================

print("\n===== INDUSTRIAL RISK ASSESSMENT =====")

for i in range(10):

    probability = y_probability[i]

    risk = get_risk_level(
        probability
    )

    print(
        f"Sample {i + 1}: "
        f"{probability * 100:.2f}% | "
        f"Risk = {risk}"
    )


# ==========================================
# 11. Save model directory
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


# ==========================================
# 12. Save trained model
# ==========================================

joblib.dump(
    model,
    "models/failure_model.pkl"
)


# ==========================================
# 13. Save scaler
# ==========================================

joblib.dump(
    scaler,
    "models/scaler.pkl"
)


print("\n===== MODEL SAVING =====")

print(
    "Model saved to: "
    "models/failure_model.pkl"
)

print(
    "Scaler saved to: "
    "models/scaler.pkl"
)
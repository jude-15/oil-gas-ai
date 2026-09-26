import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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


# ==========================================
# 4. Scale features
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. Train model
# ==========================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 6. Create SHAP explainer
# ==========================================

explainer = shap.LinearExplainer(
    model,
    X_train_scaled
)


# ==========================================
# 7. Calculate SHAP values
# ==========================================

shap_values = explainer(
    X_test_scaled
)


# ==========================================
# 8. Convert scaled data
#    back to readable feature values
# ==========================================

X_test_display = pd.DataFrame(
    X_test.values,
    columns=features
)


# ==========================================
# 9. SHAP Summary Plot
# ==========================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_display,
    show=False
)

plt.title(
    "SHAP Feature Impact - Industrial Failure Prediction"
)

plt.tight_layout()

plt.savefig(
    "data/processed/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 10. SHAP Bar Plot
# ==========================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_display,
    plot_type="bar",
    show=False
)

plt.title(
    "Feature Importance - Industrial AI"
)

plt.tight_layout()

plt.savefig(
    "data/processed/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 11. Print feature importance
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Mean Absolute SHAP": abs(
        shap_values.values
    ).mean(axis=0)
})

importance = importance.sort_values(
    by="Mean Absolute SHAP",
    ascending=False
)


print("\n===== SHAP FEATURE IMPORTANCE =====")

print(
    importance.to_string(
        index=False
    )
)


# ==========================================
# 12. Save importance results
# ==========================================

importance.to_csv(
    "data/processed/shap_feature_importance.csv",
    index=False
)


print(
    "\nSHAP analysis completed successfully!"
)

print(
    "Saved:"
)

print(
    "- data/processed/shap_summary.png"
)

print(
    "- data/processed/shap_feature_importance.png"
)

print(
    "- data/processed/shap_feature_importance.csv"
)
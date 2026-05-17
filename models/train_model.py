import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/patient_data.csv")

# =========================
# ENCODE CATEGORICAL DATA
# =========================

gender_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])

df["Disease"] = disease_encoder.fit_transform(df["Disease"])

# =========================
# FEATURES & TARGET
# =========================

X = df[
    [
        "Age",
        "Gender",
        "Heart_Rate",
        "Blood_Pressure",
        "Temperature"
    ]
]

y = df["Disease"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Model Accuracy:", accuracy)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    "models/disease_prediction_model.pkl"
)

joblib.dump(
    gender_encoder,
    "models/gender_encoder.pkl"
)

joblib.dump(
    disease_encoder,
    "models/disease_encoder.pkl"
)

print("Model saved successfully!")
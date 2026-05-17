from faker import Faker
import pandas as pd
import random

fake = Faker()

data = []

for i in range(20000):

    age = random.randint(18, 90)
    gender = random.choice(["Male", "Female"])

    disease = "Healthy"

    # =========================
    # HEALTHY
    # =========================

    heart_rate = random.randint(60, 90)
    blood_pressure = random.randint(100, 125)
    temperature = round(random.uniform(97.0, 99.0), 1)
    stay_days = random.randint(1, 3)

    # =========================
    # DISEASE ASSIGNMENT
    # =========================

    probability = random.random()

    # Heart Disease
    if age > 60 and probability > 0.75:

        disease = "Heart Disease"

        heart_rate = random.randint(95, 130)
        blood_pressure = random.randint(150, 180)
        temperature = round(random.uniform(98.0, 100.0), 1)
        stay_days = random.randint(7, 15)

    # Hypertension
    elif blood_pressure > 140 or probability > 0.82:

        disease = "Hypertension"

        heart_rate = random.randint(85, 115)
        blood_pressure = random.randint(145, 170)
        temperature = round(random.uniform(97.0, 99.5), 1)
        stay_days = random.randint(5, 10)

    # Diabetes
    elif age > 45 and probability > 0.65:

        disease = "Diabetes"

        heart_rate = random.randint(75, 105)
        blood_pressure = random.randint(120, 145)
        temperature = round(random.uniform(97.0, 99.0), 1)
        stay_days = random.randint(4, 8)

    # Flu
    elif probability > 0.88:

        disease = "Flu"

        heart_rate = random.randint(90, 120)
        blood_pressure = random.randint(105, 135)
        temperature = round(random.uniform(100.0, 104.0), 1)
        stay_days = random.randint(3, 7)

    # Asthma
    elif probability > 0.92:

        disease = "Asthma"

        heart_rate = random.randint(100, 140)
        blood_pressure = random.randint(110, 140)
        temperature = round(random.uniform(97.0, 100.0), 1)
        stay_days = random.randint(3, 9)

    # =========================
    # FINAL RECORD
    # =========================

    data.append({

        "Patient_ID": i + 1,
        "Name": fake.name(),
        "Age": age,
        "Gender": gender,
        "Disease": disease,
        "Heart_Rate": heart_rate,
        "Blood_Pressure": blood_pressure,
        "Temperature": temperature,
        "Hospital_Stay_Days": stay_days
    })

# =========================
# SAVE DATASET
# =========================

df = pd.DataFrame(data)

df.to_csv(
    "data/patient_data.csv",
    index=False
)

print(" Healthcare dataset created successfully!")
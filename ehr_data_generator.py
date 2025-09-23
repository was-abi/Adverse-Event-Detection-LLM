import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker and seed for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)

NUM_ROWS = 200  # Adjust for desired sample size

def random_sex():
    return random.choice(['M', 'F'])

def random_condition():
    return random.choice(['Hypertension', 'Diabetes', 'Asthma', 'COPD', 'None', 'Cancer'])

def random_medication():
    return random.choice(['Aspirin', 'Metformin', 'Albuterol', 'Atorvastatin', 'None'])

def random_vitals():
    return {
        "bp_sys": random.randint(90, 180),    # systolic BP
        "bp_dia": random.randint(60, 110),    # diastolic BP
        "hr": random.randint(55, 120),        # heart rate
    }

def random_note(condition, medication):
    notes = [
        f"Patient with {condition} prescribed {medication}.",
        f"Complains of chest pain after receiving {medication}.",
        "Normal exam, no adverse reaction noted.",
        f"Developed rash and swelling post {medication}.",
        "Vitals stable, no complaints today.",
        f"Elevated blood pressure, monitoring closely. On {medication}.",
        f"Reported dizziness after starting {medication}.",
        f"PT with {condition} denies new symptoms.",
    ]
    return random.choice(notes)

def random_adverse_event(note):
    # If note indicates reaction, chance of adverse event flag
    if "rash" in note or "chest pain" in note or "dizziness" in note or "swelling" in note:
        return np.random.binomial(1, 0.7)
    else:
        return np.random.binomial(1, 0.1)

data = []
for pid in range(NUM_ROWS):
    age = random.randint(18, 90)
    sex = random_sex()
    condition = random_condition()
    medication = random_medication()
    vitals = random_vitals()
    note = random_note(condition, medication)
    adverse_event = random_adverse_event(note)
    row = {
        "patient_id": pid,
        "age": age,
        "sex": sex,
        "condition": condition,
        "medication": medication,
        "bp_sys": vitals["bp_sys"],
        "bp_dia": vitals["bp_dia"],
        "heart_rate": vitals["hr"],
        "note": note,
        "adverse_event": adverse_event,
    }
    data.append(row)

# Write to CSV
df = pd.DataFrame(data)
df.to_csv('./data/synthetic_ehr.csv', index=False)

print("Sample Data:")
print(df.head())

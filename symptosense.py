import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# Load datasets
train_df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")

# Remove unnamed columns
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

# Feature and target separation
X_train = train_df.iloc[:, :-1]
y_train = train_df["prognosis"]
X_test = test_df.iloc[:, :-1]
y_test = test_df["prognosis"]

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_encoded)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# List of symptoms
symptoms_list = X_train.columns.tolist()

# Predict function for single condition
def predict_disease(input_symptoms):
    input_vector = [1 if symptom in input_symptoms else 0 for symptom in symptoms_list]
    prediction = model.predict([input_vector])[0]
    return le.inverse_transform([prediction])[0]

# Remedy database
remedy_db = {
    "Fungal infection": ("Apply antifungal cream", "Low"),
    "Allergy": ("Take antihistamines, avoid allergens", "Medium"),
    "GERD": ("Avoid spicy foods, take antacids", "Medium"),
    "Drug Reaction": ("Discontinue drug, consult a doctor immediately", "High"),
    "Diabetes": ("Control sugar, exercise, consult endocrinologist", "High"),
    "Migraine": ("Take pain relief, rest in dark room", "Medium"),
    "Hypertension": ("Reduce salt intake, regular exercise, monitor BP", "High"),
    "Common Cold": ("Stay hydrated, take steam inhalation, rest", "Low"),
    "Pneumonia": ("Antibiotics required, consult a doctor", "High"),
    "Chicken pox": ("Apply calamine lotion, avoid scratching, rest", "Medium"),
    "Dengue": ("Drink fluids, rest, consult doctor for platelet monitoring", "High"),
    "Malaria": ("Antimalarial meds required, visit a clinic", "High"),
    "Typhoid": ("Maintain hygiene, soft diet, antibiotics prescribed", "High"),
    "Hepatitis B": ("Avoid alcohol, maintain liver health, consult hepatologist", "High"),
    "Hepatitis C": ("Specialist treatment, avoid fatty foods", "High"),
    "Hepatitis D": ("Consult liver specialist, rest and hydration", "High"),
    "Hepatitis E": ("Drink clean water, light meals, rest", "Medium"),
    "Jaundice": ("Avoid oily food, take liver tonics, rest", "Medium"),
    "Tuberculosis": ("Take full course of TB meds, wear a mask", "High"),
    "Asthma": ("Use inhaler, avoid dust and allergens", "High"),
    "Bronchial Asthma": ("Inhalers, breathing exercises, avoid triggers", "High"),
    "Acne": ("Use salicylic acid-based cleanser, avoid oily food", "Low"),
    "Psoriasis": ("Apply medicated moisturizer, UV therapy", "Medium"),
    "Arthritis": ("Warm compress, anti-inflammatory meds, consult ortho", "Medium"),
    "Cervical spondylosis": ("Neck exercises, pain relief, use neck collar if advised", "Medium"),
    "Hyperthyroidism": ("Beta-blockers, monitor hormone levels", "High"),
    "Hypothyroidism": ("Take thyroxine tablets daily", "High"),
    "Urinary tract infection": ("Drink water, take prescribed antibiotics", "Medium"),
    "Varicose veins": ("Elevate legs, wear compression socks", "Medium"),
    "Vertigo": ("Lie down still, take vestibular suppressants", "Low"),
    "Paralysis (brain hemorrhage)": ("Emergency! Seek immediate medical help", "Very High"),
    "Heart attack": ("Emergency! Call ambulance immediately", "Very High"),
    "Stroke": ("Emergency! Seek immediate medical attention", "Very High"),
    "Peptic ulcer": ("Avoid spicy food, take antacids, consult gastroenterologist", "Medium")
}

# Get recommendation
def get_recommendation(disease):
    return remedy_db.get(disease, ("Consult a healthcare provider", "Medium"))

# Suggest matching symptoms
def suggest_symptoms(partial_input):
    return [symptom for symptom in symptoms_list if partial_input.lower() in symptom.lower()]

# Multi-label model
mlb = MultiLabelBinarizer()
y_train_multilabel = mlb.fit_transform([[d] for d in y_train])
y_test_multilabel = mlb.transform([[d] for d in y_test])

multi_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
multi_model.fit(X_train, y_train_multilabel)

# Multi-condition prediction
def predict_possible_diseases(input_symptoms):
    input_vector = [1 if symptom in input_symptoms else 0 for symptom in symptoms_list]
    prediction = multi_model.predict([input_vector])[0]
    return list(mlb.inverse_transform([prediction])[0])

# Graphical output
def plot_predictions(disease_list):
    remedies = [get_recommendation(d)[0] for d in disease_list]
    urgency = [get_recommendation(d)[1] for d in disease_list]
    colors = {"Low": "green", "Medium": "orange", "High": "red", "Very High": "darkred"}

    plt.figure(figsize=(10, 6))
    bars = plt.barh(disease_list, [1]*len(disease_list), color=[colors[u] for u in urgency])
    for bar, remedy in zip(bars, remedies):
        plt.text(1.02, bar.get_y() + bar.get_height()/2, remedy, va='center')

    plt.title("Predicted Health Concerns with Remedies")
    plt.xlabel("Urgency Levels")
    plt.xticks([])
    plt.tight_layout()
    plt.show()

# Example usage
input_symptoms = ["fatigue", "vomiting", "headache"]
predicted_diseases = predict_possible_diseases(input_symptoms)

print("Possible diseases from these symptoms:")
for dis in predicted_diseases:
    remedy, urgency = get_recommendation(dis)
    print(f" - {dis} | Remedy: {remedy} | Urgency: {urgency}")

plot_predictions(predicted_diseases)

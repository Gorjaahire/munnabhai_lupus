import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("hospital_dataset.csv")

# Features (input data)
X = df[["ER_patients", "Scheduled_surgeries", "current_beds_full", "ICU_full", "staff_on_duty"]]

# Target (what we predict)
y = df["current_beds_full"]  # We train the model to predict bed usage

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose ML model
model = RandomForestRegressor(n_estimators=200, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
with open("bed_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as bed_model.pkl")

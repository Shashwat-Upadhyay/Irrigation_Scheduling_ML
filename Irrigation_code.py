import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError # Import the MeanSquaredError class
from sklearn.preprocessing import MinMaxScaler
import os

# Load dataset
import pandas as pd
file_path = "temperature_dataset.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns & handle missing values
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df = df.dropna()

# Feature selection
X = df[['T_max (°C)', 'T_min (°C)']].values

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load trained CNN model
MODEL_PATH = "irrigation_cnn_model.h5"

if not os.path.exists(MODEL_PATH):
    print("🚨 Model file not found! Train & save 'irrigation_cnn_model.h5' first.")
    exit()

# Specify custom_objects when loading the model, forcing Keras to recognize mse
model = load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()}) # Pass the MeanSquaredError class to custom_objects

def predict_irrigation(T_max, T_min):
    """Predict ET₀ and irrigation duration based on temperature inputs."""
    user_input = np.array([[T_max, T_min]])
    user_input_scaled = scaler.transform(user_input)
    user_input_scaled = user_input_scaled.reshape((user_input_scaled.shape[0], user_input_scaled.shape[1], 1))

    # Predict ET₀ using CNN model
    predicted_et0 = model.predict(user_input_scaled)[0][0]

    # Calculate irrigation duration
    crop_factor = 1.1  # Example crop coefficient for wheat
    area_factor = 1.0  # Example irrigation area factor
    irrigation_duration = predicted_et0 * crop_factor * area_factor

    # Display results
    print(f"\n🌊 **Predicted ET₀:** {predicted_et0:.3f} mm/day")
    print(f"⏳ **Recommended Irrigation Duration:** {irrigation_duration:.2f} minutes/day")

# User Input (Loop for Continuous Predictions)
while True:
    try:
        T_max = float(input("Enter Max Temperature (°C): "))
        T_min = float(input("Enter Min Temperature (°C): "))
        predict_irrigation(T_max, T_min)

        # Ask user if they want another prediction
        cont = input("Do you want to predict again? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Exiting program. Have a great day! 😊")
            break
    except ValueError:
        print("❌ Invalid input! Please enter numerical values only.")


import pickle
from sklearn.preprocessing import MinMaxScaler

# Fit the scaler
scaler = MinMaxScaler()
scaler.fit(X)

# Save the scaler as a .pkl file
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Scaler saved as 'scaler.pkl'")
#Load the Scaler from .pkl File in the Future
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Now transform the data
X_scaled = scaler.transform(X)
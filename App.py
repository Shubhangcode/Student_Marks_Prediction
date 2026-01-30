
import streamlit as st
import numpy as np
import pip
import pyngrok
import pandas as pd

from Model_Training import X_test

df = pd.read_csv("data/processed/processed_data.csv")

import joblib
import warnings
warnings.filterwarnings("ignore")
joblib.load("best_model.pkl").predict(X_test)
# Load the trained model
model = joblib.load('best_model.pkl')

st.title('Student Exam Score Prediction')
study_hours= st.slider("Study Hours per Day", 0.0, 12.0, 2.0)
attendence= st.slider("Attendence Percentage", 0.0, 100.0, 80.0)
sleep_hours= st.slider("Sleep Hours", 0.0, 12.0, 7.0)
mental_health= st.slider("Mental Health Rating", 1.0, 10.0, 5.0)
part_time_job= st.selectbox("Part Time Job", ["Yes", "No"])
ptj_encoded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Exam Score"):
    input_data = np.array([[study_hours,attendence,sleep_hours,mental_health,ptj_encoded]])
    prediction = model.predict(input_data)[0]

    prediction = max(0,min(100,prediction))
    st.success(f'Predicted Exam Score: {prediction:.2f}')

import os
import subprocess


from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokError # Import PyngrokNgrokError correctly

# Kill any existing ngrok processes to ensure a clean start
ngrok.kill()

# Prompt the user for their ngrok authtoken
# You can get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
print("Please enter your ngrok authtoken (it should start with 'authtoken_'):")
NGROK_AUTH_TOKEN = input().strip()

if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    print("Ngrok authtoken set successfully.")
else:
    print("No ngrok authtoken provided. ngrok will not function correctly.")

public_url = None
try:
    # Connect ngrok to port 8501 where Streamlit app will run
    public_url = ngrok.connect(8501)
    print(f"Streamlit App URL: {public_url}")

    
    subprocess.Popen(["streamlit", "run", "app.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except PyngrokNgrokError as e: # Use the correctly imported exception
    print(f"Error connecting to ngrok: {e}")
    print("Please ensure your ngrok authtoken is correct and try again. It typically starts with 'authtoken_'.")
    print("You can find your authtoken at https://dashboard.ngrok.com/get-started/your-authtoken")
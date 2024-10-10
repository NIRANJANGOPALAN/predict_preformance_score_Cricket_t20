import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import joblib

# Load the trained model
model = joblib.load('xgb_model.joblib')

# Load the scaler
scaler = joblib.load('scaler.joblib')

# Load the label encoder
le = joblib.load('label_encoder.joblib')

st.title('Cricket Player Performance Predictor')

st.write("""
This app predicts a cricket player's performance based on their statistics.
Please enter the player's details below:
""")

# Input fields
striker = st.text_input('Player Name')
player_type = st.selectbox('Player Type', ['Batsman', 'Bowler', 'All-rounder'])
total_runs_scored = st.number_input('Total Runs Scored', min_value=0)
total_batting_average = st.number_input('Total Batting Average', min_value=0.0)
batting_strike_rate = st.number_input('Batting Strike Rate', min_value=0.0)
total_wickets = st.number_input('Total Wickets', min_value=0)
economy_rate = st.number_input('Economy Rate', min_value=0.0)
total_balls_faced = st.number_input('Total Balls Faced', min_value=0)
balls_bowled = st.number_input('Balls Bowled', min_value=0)
overs_bowled_clean = st.number_input('Overs Bowled (Clean)', min_value=0.0)

if st.button('Predict Performance'):
    # Create a dataframe with the input
    input_data = pd.DataFrame({
        'Player_type': [player_type],
        'totalrunsscored': [total_runs_scored],
        'Total_batting_average': [total_batting_average],
        'batting_strike_rate': [batting_strike_rate],
        'totalwickets': [total_wickets],
        'economyrate': [economy_rate],
        'totalballsfaced': [total_balls_faced],
        'Balls Bowled': [balls_bowled],
        'oversbowled_clean': [overs_bowled_clean]
    })

    # Encode the Player_type
    input_data['Player_type'] = le.transform(input_data['Player_type'])

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    st.write(f"Predicted Performance Score for {striker}: {prediction[0]:.2f}")

st.write("""
### Note:
- The model uses advanced machine learning techniques to predict player performance.
- The prediction is based on the statistics provided and may not account for all factors that influence a player's performance.
- Use this prediction as a guide, not as a definitive measure of a player's abilities.
""")
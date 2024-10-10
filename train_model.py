import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

# Sample data
data = [
    ['Jos Buttler', 'Batsman', 391, 43.44, 158.62, 0, 0, 246, 0, 0],
    ['Tymal Mills', 'Bowler', 15, 7.5, 125, 16, 8.2, 20, 120, 20],
    ['Will Jacks', 'All-rounder', 230, 32.86, 145.57, 3, 7.8, 158, 30, 5],
    ['Liam Livingstone', 'All-rounder', 185, 26.43, 152.89, 5, 8.5, 140, 40, 6.4],
    ['Reece Topley', 'Bowler', 20, 10, 111.11, 11, 7.9, 25, 90, 15],
    ['Dawid Malan', 'Batsman', 278, 39.71, 140.4, 0, 0, 180, 0, 0],
    ['Sam Curran', 'All-rounder', 160, 22.86, 133.33, 8, 8.7, 130, 70, 11.4],
    ['Tom Abell', 'Batsman', 145, 24.17, 128.32, 2, 9.2, 120, 20, 3.2],
    ['Adil Rashid', 'Bowler', 35, 11.67, 106.06, 10, 7.5, 30, 80, 13.2],
    ['Harry Brook', 'Batsman', 238, 47.6, 172.46, 0, 0, 150, 0, 0]
]

# Create DataFrame
columns = ['striker', 'Player_type', 'totalrunsscored', 'Total_batting_average', 'batting_strike_rate',
           'totalwickets', 'economyrate', 'totalballsfaced', 'Balls Bowled', 'oversbowled_clean']
df = pd.DataFrame(data, columns=columns)

# Add a dummy 'Overall_score' column (you may want to replace this with actual scores if available)
df['Overall_score'] = np.random.uniform(60, 100, len(df))

# Prepare the features and target
X = df.drop(['striker', 'Overall_score'], axis=1)
y = df['Overall_score']

# Encode categorical variables
le = LabelEncoder()
X['Player_type'] = le.fit_transform(X['Player_type'])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(xgb_model, 'xgb_model.joblib')

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Save the label encoder
joblib.dump(le, 'label_encoder.joblib')

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
print(f"Accuracy: {r2 * 100:.2f}%")
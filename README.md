# =====================================================
# MAVIN: Intelligent Traffic Lane Recommendation System
# Single-file Prototype
# =====================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------
df = pd.read_csv("Mavin_Traffic_With_Lanes.csv")

# -----------------------------------------------------
# 2. ADD PROTOTYPE USER LOCATIONS
# -----------------------------------------------------
# Simulated real-world user locations
USER_LOCATIONS = ["City Center", "Airport Road", "Market District", "Residential Zone"]

np.random.seed(42)
df["user_location"] = np.random.choice(USER_LOCATIONS, size=len(df))

# -----------------------------------------------------
# 3. ENCODE CATEGORICAL FEATURES
# -----------------------------------------------------
encoders = {}

categorical_columns = [
    "lane",
    "area",
    "Day of the week",
    "user_location"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------------------------------
# 4. FEATURE SELECTION
# -----------------------------------------------------
FEATURES = [
    "hour",
    "lane",
    "area",
    "user_location",
    "CarCount",
    "BikeCount",
    "BusCount",
    "TruckCount"
]

X = df[FEATURES]
y = df["traffic_volume"]

# -----------------------------------------------------
# 5. TRAIN MODEL
# -----------------------------------------------------
model = RandomForestRegressor(
    n_estimators=250,
    random_state=42
)

model.fit(X, y)

# -----------------------------------------------------
# 6. HELPER FUNCTIONS
# -----------------------------------------------------
def traffic_status(volume):
    if volume < 40:
        return "Light"
    elif volume < 80:
        return "Moderate"
    return "Heavy"

def car_percentage_per_lane():
    totals = df.groupby("lane")["CarCount"].sum()
    total_cars = totals.sum()
    percentages = (totals / total_cars) * 100
    return percentages

# -----------------------------------------------------
# 7. LANE RECOMMENDATION ENGINE
# -----------------------------------------------------
def recommend_lane(hour, area_name, location_name):

    area_encoded = encoders["area"].transform([area_name])[0]
    location_encoded = encoders["user_location"].transform([location_name])[0]

    lane_names = encoders["lane"].classes_
    results = []

    for lane in lane_names:
        lane_encoded = encoders["lane"].transform([lane])[0]

        lane_data = df[df["lane"] == lane_encoded]

        sample = pd.DataFrame([{
            "hour": hour,
            "lane": lane_encoded,
            "area": area_encoded,
            "user_location": location_encoded,
            "CarCount": lane_data["CarCount"].mean(),
            "BikeCount": lane_data["BikeCount"].mean(),
            "BusCount": lane_data["BusCount"].mean(),
            "TruckCount": lane_data["TruckCount"].mean()
        }])

        predicted_volume = model.predict(sample)[0]

        results.append({
            "Lane": lane,
            "Predicted Volume": round(predicted_volume, 1),
            "Traffic Status": traffic_status(predicted_volume)
        })

    results_df = pd.DataFrame(results).sort_values("Predicted Volume")
    return results_df

# -----------------------------------------------------
# 8. USER INTERACTION (CLI PROTOTYPE)
# -----------------------------------------------------
if __name__ == "__main__":

    print("\n🚦 WELCOME TO MAVIN 🚦")
    print("Intelligent Traffic Lane Recommendation System\n")

    # ---- Time Input ----
    hour = int(input("Enter current hour (0 - 23): "))

    # ---- Location Input ----
    print("\nSelect your current location:")
    for i, loc in enumerate(USER_LOCATIONS, 1):
        print(f"{i}. {loc}")

    location_choice = int(input("\nEnter choice (1-4): "))
    user_location = USER_LOCATIONS[location_choice - 1]

    # ---- Area Input ----
    AREAS = encoders["area"].classes_
    print("\nSelect your area:")
    for i, area in enumerate(AREAS, 1):
        print(f"{i}. {area}")

    area_choice = int(input("\nEnter choice: "))
    user_area = AREAS[area_choice - 1]

    # ---- Prediction ----
    recommendations = recommend_lane(hour, user_area, user_location)

    best_lane = recommendations.iloc[0]

    print("\n🚥 MAVIN DECISION OUTPUT 🚥")
    print("----------------------------------")
    print(f"Time: {hour}:00")
    print(f"Area: {user_area}")
    print(f"Location: {user_location}")

    print(f"\n✅ Recommended Lane: {best_lane['Lane']}")
    print(f"Reason: Lowest predicted congestion")
    print(f"Predicted Traffic Volume: {best_lane['Predicted Volume']}")

    print("\n📊 Traffic on Other Lanes:")
    for _, row in recommendations.iterrows():
        print(f"- {row['Lane']}: {row['Traffic Status']}")

    # ---- Car Distribution ----
    print("\n🚗 Percentage of Cars per Lane:")
    car_percentages = car_percentage_per_lane()

    for lane_encoded, percent in car_percentages.items():
        lane_name = encoders["lane"].inverse_transform([lane_encoded])[0]
        print(f"- {lane_name}: {percent:.2f}%")


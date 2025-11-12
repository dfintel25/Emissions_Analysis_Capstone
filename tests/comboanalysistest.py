# Core libraries
import pandas as pd
import numpy as np
import sqlite3
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning (we’ll use later)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Change to project root (adjust as needed)
os.chdir("C:/Projects/Emissions_Analysis_Capstone")

db_path = os.path.join('emissions_data', 'emissionsdata.sqlite')
os.makedirs(os.path.dirname(db_path), exist_ok=True)
conn = sqlite3.connect(db_path)

# === Livestock query (filtered) ===
livestock_query = """
SELECT
    region,
    animal_type,
    production_system,
    commodity,
    co2_emissions_kg,
    n2o_emissions_kg
FROM livestock_emissions
WHERE region = 'North America'
  AND animal_type = 'Cattle'
  AND production_system = 'Aggregated'
  AND commodity = 'Meat';
"""
livestock_df = pd.read_sql(livestock_query, conn)

'''
# Filter livestock to North America, cattle, meat commodity
cattle_meat_na_df = livestock_df[
    (livestock_df["region"] == "North America") &
    (livestock_df["animal_type"] == "Cattle") &
    (livestock_df["commodity"] == "Meat")
].copy()
'''

# Calculate total cattle CO2 emissions
cattle_total_CO2_kg = cattle_meat_na_df["co2_emissions_kg"].sum()
cattle_total_CO2_million_tonnes = cattle_total_CO2_kg / 1e6

# === Vehicle query ===
vehicle_query = """
SELECT
    make,
    model,
    co2_emissions_g_per_km,
    fuel_type
FROM vehicle_emissions;
"""
vehicle_df = pd.read_sql(vehicle_query, conn)

# Convert CO2 column to numeric
vehicle_df["co2_emissions_g_per_km"] = pd.to_numeric(
    vehicle_df["co2_emissions_g_per_km"], errors="coerce"
)

# Convert g/km → kg/km
vehicle_df["co2_emissions_kg_per_km"] = vehicle_df["co2_emissions_g_per_km"] / 1000

# Multiply by average annual mileage to get kg/year
AVG_KM_PER_YEAR = 16094  # 10,000 miles
vehicle_df["co2_emissions_kg_per_year"] = vehicle_df["co2_emissions_kg_per_km"] * AVG_KM_PER_YEAR

# Dataset aggregates 7 years, divide by NUM_YEARS to get avg per year
NUM_YEARS = 7
vehicle_df["co2_emissions_kg_per_year"] = vehicle_df["co2_emissions_kg_per_year"] / NUM_YEARS

# Total vehicles sold in 2015 (NA)
TOTAL_VEHICLES_2015 = 17_470_499
vehicle_total_CO2_kg_2015 = vehicle_df["co2_emissions_kg_per_year"].mean() * TOTAL_VEHICLES_2015
vehicle_total_CO2_million_tonnes = vehicle_total_CO2_kg_2015 / 1e6

# CO2 per vehicle per year
vehicle_CO2_per_vehicle_kg_per_year = vehicle_df["co2_emissions_kg_per_year"].mean()

# === Summary comparison (totals + per-vehicle) ===
summary_df = pd.DataFrame({
    "Category": ["Cattle Meat (NA)", "Vehicles (NA, 2015 sales)"],
    "Total_CO2_kg_per_year": [cattle_total_CO2_kg, vehicle_total_CO2_kg_2015],
    "Total_CO2_million_tonnes_per_year": [cattle_total_CO2_million_tonnes, vehicle_total_CO2_million_tonnes],
    "Per_Unit_CO2_kg_per_year": [None, vehicle_CO2_per_vehicle_kg_per_year]
})

print("CO₂ Emissions Comparison (Total + Per Vehicle):")
display(summary_df)

# === Visualization: Total emissions ===
plt.figure(figsize=(8,6))
sns.barplot(
    x="Category",
    y="Total_CO2_million_tonnes_per_year",
    data=summary_df,
    palette=["saddlebrown", "steelblue"]
)
plt.ylabel("CO₂ Emissions (Million Tonnes/Year)")
plt.title("CO₂ Emissions: Cattle Meat vs Vehicles (North America)")
plt.xticks(rotation=15)
plt.show()

# === Visualization: Per-vehicle emissions ===
plt.figure(figsize=(8,6))
sns.barplot(
    x="Category",
    y="Per_Unit_CO2_kg_per_year",
    data=summary_df,
    palette=["saddlebrown", "steelblue"]
)
plt.ylabel("CO₂ Emissions per Unit (kg/year)")
plt.title("CO₂ Emissions per Vehicle (Cattle per-animal data not available)")
plt.xticks(rotation=15)
plt.show()

# --- Livestock CO2 statistics ---
print("=== Livestock CO2 Emissions (kg/year) ===")
livestock_stats = livestock_df["co2_emissions_kg"].describe()
display(livestock_stats)

# --- Vehicle CO2 statistics ---
print("\n=== Vehicle CO2 Emissions (kg/year) ===")
vehicle_stats = vehicle_df["co2_emissions_kg_per_year"].describe()
display(vehicle_stats)

plt.figure(figsize=(8,6))
sns.barplot(
    x="Category",
    y="Total_CO2_million_tonnes_per_year",
    data=summary_df,
    palette=["saddlebrown", "steelblue"]
)
plt.ylabel("CO₂ Emissions (Million Tonnes/Year)")
plt.title("Total CO₂ Emissions: Cattle Meat vs Vehicles (North America)")
plt.xticks(rotation=15)
for i, v in enumerate(summary_df["Total_CO2_million_tonnes_per_year"]):
    plt.text(i, v*1.01, f"{v:,.0f}", ha='center', fontweight='bold')
plt.show()

plt.figure(figsize=(6,6))
sns.barplot(
    x=["Vehicle (avg CO₂ per unit/year)"],
    y=[vehicle_CO2_per_vehicle_kg_per_year],
    palette=["steelblue"]
)
plt.ylabel("CO₂ Emissions per Vehicle (kg/year)")
plt.title("Average Per-Vehicle CO₂ Emissions (NA, 2015)")
plt.text(0, vehicle_CO2_per_vehicle_kg_per_year*1.01, f"{vehicle_CO2_per_vehicle_kg_per_year:,.0f}", ha='center', fontweight='bold')
plt.show()

plt.figure(figsize=(8,6))
totals = [cattle_total_CO2_million_tonnes, vehicle_total_CO2_million_tonnes]
labels = ["Cattle Meat (NA)", "Vehicles (NA, 2015 sales)"]
colors = ["saddlebrown", "steelblue"]

plt.bar(labels, totals, color=colors)
plt.ylabel("CO₂ Emissions (Million Tonnes/Year)")
plt.title("Total CO₂ Emissions: Livestock vs Vehicles")
plt.yscale("log")  # optional, log scale to handle huge differences
for i, v in enumerate(totals):
    plt.text(i, v*1.05, f"{v:,.0f}", ha='center', fontweight='bold')
plt.show()

plt.figure(figsize=(6,6))
plt.pie(
    totals,
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=(0.05, 0.05)
)
plt.title("Proportion of Total CO₂ Emissions (NA)")
plt.show()

# --- Encode categorical variables ---
vehicle_model_df = vehicle_df.copy()
vehicle_model_df = pd.get_dummies(vehicle_model_df, columns=["fuel_type", "make"], drop_first=True)

# --- Define features and target ---
X = vehicle_model_df.drop(columns=["model", "co2_emissions_g_per_km", "co2_emissions_kg_per_km", "co2_emissions_kg_per_year"])
y = vehicle_model_df["co2_emissions_g_per_km"]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Fit Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)

# --- Predict ---
y_pred = lr.predict(X_test)

# --- Evaluate ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression RMSE: {rmse:.2f}")
print(f"Linear Regression R²: {r2:.2f}")

# Assume vehicle_model_df is already one-hot encoded
X = vehicle_model_df.drop(columns=["model", "co2_emissions_g_per_km",
                                   "co2_emissions_kg_per_km", "co2_emissions_kg_per_year"])
y = vehicle_model_df["co2_emissions_g_per_km"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Random Forest RMSE: {rmse:.2f}")
print(f"Random Forest R²: {r2:.2f}")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importances.head(10)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,6))
top_features.plot(kind='barh', color='steelblue')
plt.gca().invert_yaxis()  # Highest importance on top
plt.xlabel("Feature Importance")
plt.title("Top 10 Features Predicting CO₂ Emissions")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual CO₂ g/km")
plt.ylabel("Predicted CO₂ g/km")
plt.title("Actual vs Predicted CO₂ Emissions")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='darkorange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual CO₂ g/km")
plt.ylabel("Predicted CO₂ g/km")
plt.title("Random Forest: Actual vs Predicted CO₂ Emissions")
plt.show()

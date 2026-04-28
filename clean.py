import pandas as pd

# 1. Load dataset
df = pd.read_csv("clean_weather.csv")

# 2. Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# 3. Rename column to 'weather' (since your column is 'Weather')
df.rename(columns={"Weather": "weather"}, inplace=True)

# 4. Define mapping function
def map_weather(x):
    x = str(x).lower()

    if "sun" in x :
        return "Sunny"
    elif "cloud" in x or "overcast" in x:
        return "Cloudy"
    elif "rain" in x :
        return "Rainy"
    elif "fog" in x :
        return "Foggy"
    else:
        return None  # remove unwanted classes

# 5. Apply mapping
df["weather"] = df["weather"].apply(map_weather)

# 6. Remove rows with unknown classes
df = df.dropna(subset=["weather"])

# 7. Save cleaned dataset (NEW FILE)
df.to_csv("final_weather_4class.csv", index=False)

# 8. Verify output
print("✅ Cleaning done!")
print("Saved as: Cleaned_data.csv")
print("\nClass distribution:")
print(df["weather"].value_counts())
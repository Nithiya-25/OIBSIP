import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"C:\Users\NITHIYA SRI G\OneDrive\Documents\Desktop\OIBSIP\Unemployment_ in_ India.csv", encoding="latin1")

# Clean columns
data.columns = data.columns.str.strip()

# Convert Date column
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# ===============================
# Graph 1: Overall Unemployment
# ===============================
plt.figure(figsize=(10,5))
plt.plot(data['Date'], data['Estimated Unemployment Rate (%)'])
plt.title("Unemployment Rate in India Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# ===============================
# Graph 2: Rural vs Urban
# ===============================
plt.figure(figsize=(10,5))

for area in data['Area'].unique():
    subset = data[data['Area'] == area]
    plt.plot(subset['Date'], subset['Estimated Unemployment Rate (%)'], label=area)

plt.title("Unemployment Rate: Rural vs Urban")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()
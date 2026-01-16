import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 1. Load Dataset
df = pd.read_csv("Advertising.csv")

# Remove blank column if exists
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
print("\nDataset Preview:")
print(df.head())

# 2. Define Features & Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Model Evaluation
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 7. VISUALIZATIONS

# --- Graph 1: TV vs Sales ---
plt.figure()
plt.scatter(df['TV'], df['Sales'])
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")
plt.show()

# --- Graph 2: Radio vs Sales ---
plt.figure()
plt.scatter(df['Radio'], df['Sales'])
plt.xlabel("Radio Advertising Budget")
plt.ylabel("Sales")
plt.title("Radio Advertising vs Sales")
plt.show()

# --- Graph 3: Newspaper vs Sales ---
plt.figure()
plt.scatter(df['Newspaper'], df['Sales'])
plt.xlabel("Newspaper Advertising Budget")
plt.ylabel("Sales")
plt.title("Newspaper Advertising vs Sales")
plt.show()

# --- Graph 4: Actual vs Predicted Sales ---
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.show()

# 8. User Input Prediction
print("\n---- Sales Prediction (User Input) ----")
tv = float(input("Enter TV Advertising Budget: "))
radio = float(input("Enter Radio Advertising Budget: "))
newspaper = float(input("Enter Newspaper Advertising Budget: "))
user_input = pd.DataFrame([[tv, radio, newspaper]],columns=['TV', 'Radio', 'Newspaper'])
predicted_sales = model.predict(user_input)
print(f"Predicted Sales: {predicted_sales[0]:.2f}")
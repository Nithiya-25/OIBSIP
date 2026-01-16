import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 1. Load dataset
data = pd.read_csv(r"C:\Users\NITHIYA SRI G\OneDrive\Documents\Desktop\OIBSIP\car_data.csv")
print("Dataset Preview:")
print(data.head())
# 2. Convert categorical columns into numbers
data = pd.get_dummies(data, drop_first=True)
# 3. Split features and target
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]
# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# 6. Make predictions
y_pred = model.predict(X_test)
# 7. Model evaluation
print("\nModel Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
# 8. Visualization: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(y_test,y_pred)
plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Actual vs Predicted Car Price")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Month": [1,2,3,4,5,6],
    "Sales": [100,120,130,150,170,200]
}

df = pd.DataFrame(data)

# Train model
X = df[["Month"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

# Predict future
future_months = [[7],[8],[9]]
predictions = model.predict(future_months)

print("Future Sales:", predictions)

# Plot
plt.scatter(df["Month"], df["Sales"])
plt.plot(df["Month"], model.predict(X))
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecast")
plt.show()
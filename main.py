import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. Load Data (California Housing)
url = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
df = pd.read_csv(url)

# Rename and Scale Target Variable
df.rename(columns={'median_house_value': 'Price'}, inplace=True)
df['Price'] = df['Price'] / 100000  # Scaling for readability

# 2. Split Data
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Baseline Model: Linear Regression
# Assumes a straight-line relationship (often too simple)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)
lin_r2 = r2_score(y_test, lin_pred)

# 4. Advanced Model: Random Forest
# Captures non-linear patterns and complex interactions
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)

# 5. Performance Comparison
print(f"📊 Linear Regression R2: {lin_r2:.2f}")
print(f"🌲 Random Forest R2:    {rf_r2:.2f}")
improvement = ((rf_r2 - lin_r2) / lin_r2) * 100
print(f"🚀 Improvement: {improvement:.1f}%")

# 6. Visualization & Save
plt.figure(figsize=(12, 6))

# Plot Linear Regression Results
plt.subplot(1, 2, 1)
plt.scatter(y_test, lin_pred, alpha=0.3, color='#3498db') # Blue
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'Linear Regression\nR2 Score: {lin_r2:.2f}')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

# Plot Random Forest Results
plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_pred, alpha=0.3, color='#2ecc71') # Green
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'Random Forest (Winner)\nR2 Score: {rf_r2:.2f}')
plt.xlabel('Actual Price')

plt.tight_layout()

# Save the chart as high-quality PNG
plt.savefig('model_comparison.png', dpi=300)
print("✅ Chart saved as 'model_comparison.png'")

plt.show()
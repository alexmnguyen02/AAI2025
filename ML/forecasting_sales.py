import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dates = pd.date_range(start="2020-01-01", periods=36, freq='M')
demand = np.linspace(200, 400, 36) + np.random.normal(0, 15, 36)

df = pd.DataFrame({
    'date': dates,
    'demand': demand
})

df['month_num'] = np.arange(len(df))

X = df[['month_num']]
y = df['demand']

model = LinearRegression()
model.fit(X, y)

# Forecast next 6 months
future_months = np.arange(len(df), len(df)+6).reshape(-1,1)
forecast = model.predict(future_months)

future_dates = pd.date_range(start=df['date'].iloc[-1], periods=7, freq='M')[1:]

# Plot
plt.plot(df['date'], df['demand'], label="Historical")
plt.plot(future_dates, forecast, label="Forecast", linestyle='--')
plt.legend()
plt.title("Housing Demand Forecast (Next 6 Months)")
plt.show()

print("Forecasted Demand:", forecast)

# Load dataset
df = pd.read_csv('sales_data.csv')  # Ensure columns: 'month', 'sales'
X = df[['month']]
y = df['sales']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict for next 6 months
future_months = pd.DataFrame({'month': range(max(df['month'])+1, max(df['month'])+7)})
predictions = model.predict(future_months)

# Plot results
plt.plot(df['month'], y, label='Historical Sales')
plt.plot(future_months['month'], predictions, label='Predicted Sales', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()

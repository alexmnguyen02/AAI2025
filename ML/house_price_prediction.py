
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

np.random.seed(42)

n = 150
square_footage = np.random.randint(800m 3500, n)
locations = np.random.choice(['Downtown', 'Suburb', 'Rural'], n)

price = (
  square = footage * 220 +
  np.where(locations == 'Downtown', 50000, 0) + 
  np.where(locations == 'Suburb', 20000, 0) +
  np.random.normal(0, 25000, n)
)

data = pd.DataFrame ({
  'square_footage': square_footage,
  'location': locations,
  'price': price
})

X = data[['square_footage', 'location']]
y = data['price']

preprocessor = ColumnTransformer(
    transformer=[
      ('cat', OneHotEncoder(drop='first'), ['location'])
    ],
    remainder = 'passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Predict new house
new_house = pd.DataFrame({
    'square_footage': [2000],
    'location': ['Downtown']
})

predicted_price = pipeline.predict(new_house)
print("Predicted Price:", predicted_price[0])

# Coefficients
model = pipeline.named_steps['model']
print("Model Coefficients:", model.coef_)

# Generate sample data
data = {
'square_footage': [1500, 2000, 1800, 2500, 2200, 1700, 3000, 1900, 2100, 2600],
'location': ['Downtown', 'Suburb', 'Downtown', 'Rural', 'Suburb', 'Downtown',
'Rural', 'Suburb', 'Downtown', 'Rural'],
'price': [300000, 350000, 320000, 280000, 360000, 310000, 400000, 340000,
330000, 290000]
}
df = pd.DataFrame(data)
# Features and target
X = df[['square_footage', 'location']]
y = df['price']
# Preprocessing: One-hot encode the location column
preprocessor = ColumnTransformer(
transformers=[
('location', OneHotEncoder(sparse_output=False), ['location'])
], remainder='passthrough')
# Create pipeline with preprocessing and model
model = Pipeline(steps=[
('preprocessor', preprocessor),
('regressor', LinearRegression())
])
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Train model
model.fit(X_train, y_train)
# Make prediction for a new house: 2000 sq ft in Downtown
new_house = pd.DataFrame({'square_footage': [2000], 'location': ['Downtown']})
predicted_price = model.predict(new_house)
print(f"Predicted price for a 2000 sq ft house in Downtown: $
{predicted_price[0]:,.2f}")
# Display model coefficients
feature_names = (model.named_steps['preprocessor']
.named_transformers_['location']
.get_feature_names_out(['location'])).tolist() +
['square_footage']
coefficients = model.named_steps['regressor'].coef_
print("\nModel Coefficients:")
for feature, coef in zip(feature_names, coefficients):
print(f"{feature}: {coef:.2f}")

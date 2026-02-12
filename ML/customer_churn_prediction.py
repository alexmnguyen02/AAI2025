import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

np.random.seed(42)

n = 200

data = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'monthly_usage': np.random.randint(10, 100, n),
    'purchase_amount': np.random.randint(20, 500, n),
    'customer_service_calls': np.random.randint(0, 10, n),
    'region': np.random.choice(['East', 'West', 'North', 'South'], n)
})

# Churn logic
data['churn'] = (
    (data['customer_service_calls'] > 5) |
    (data['monthly_usage'] < 30)
).astype(int)

X = data.drop('churn', axis=1)
y = data['churn']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age','monthly_usage','purchase_amount','customer_service_calls']),
    ('cat', OneHotEncoder(drop='first'), ['region'])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict new customer
new_customer = pd.DataFrame({
    'age':[45],
    'monthly_usage':[20],
    'purchase_amount':[100],
    'customer_service_calls':[7],
    'region':['East']
})

prob = pipeline.predict_proba(new_customer)[0][1]
print("Churn Probability:", prob)
print("At Risk?" , prob > 0.5)

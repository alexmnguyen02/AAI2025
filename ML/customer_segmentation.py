import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)

n = 180

data = pd.DataFrame({
    'annual_spending': np.random.randint(500, 20000, n),
    'purchase_frequency': np.random.randint(1, 50, n),
    'age': np.random.randint(18, 70, n)
})

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Elbow method
inertia = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,6), inertia)
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Apply K=3
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

print(data.groupby('Cluster').mean())

data.to_csv("customer_segments.csv", index=False)

# Generate sample customer data
data = {
'annual_spending': [500, 1200, 300, 1500, 800, 200, 1000, 600, 1300, 400],
'purchase_frequency': [5, 12, 3, 15, 8, 2, 10, 6, 13, 4],
'age': [25, 34, 45, 28, 52, 36, 41, 29, 47, 33],
'region': ['North', 'South', 'West', 'East', 'South', 'North', 'West', 'East',
'South', 'North']
}
df = pd.DataFrame(data)
# Preprocess data: Select numerical features and scale them
features = ['annual_spending', 'purchase_frequency', 'age']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Determine optimal number of clusters using elbow method
inertia = []
K = range(1, 6)
for k in K:
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
inertia.append(kmeans.inertia_)
# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.savefig('elbow_plot.png')
plt.close()
# Apply K-Means with optimal K (e.g., 3 based on elbow method)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
# Analyze clusters
cluster_summary = df.groupby('cluster')[features].mean().round(2)
print("Cluster Characteristics:")
print(cluster_summary)
# Example of targeted strategies
for cluster in range(optimal_k):
print(f"\nCluster {cluster} Strategy:")
if cluster_summary.loc[cluster, 'annual_spending'] > 1000:
print("High-spending customers: Offer exclusive promotions or loyalty
rewards.")
elif cluster_summary.loc[cluster, 'purchase_frequency'] > 10:
print("Frequent buyers: Provide bulk discounts or subscription plans.")
else:
print("Low-engagement customers: Send personalized re-engagement
campaigns.")
# Save cluster assignments to CSV
df.to_csv('customer_segments.csv', index=False)

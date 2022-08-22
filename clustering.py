import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import load_diabetes
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

diabetes = load_diabetes()
x = diabetes["data"]

# K Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x)
kmeans.labels_

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(x)

# DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=6)
dbscan.fit(x)

# Combine
df = pd.DataFrame(x, columns=diabetes["feature_names"])
df["kmeans"] = kmeans.labels_
df["agg"] = agg.labels_
df["dbscan"] = dbscan.labels_ #-1 means noise - nothing to do with other clusters

# Pairplot
cols = ["age", "bmi", "bp", "agg"]
sns.pairplot(df[cols], hue="agg")

# Evaluation
print(calinski_harabasz_score(x, kmeans.labels_))
print(calinski_harabasz_score(x, agg.labels_))
print(calinski_harabasz_score(x, dbscan.labels_))

print(davies_bouldin_score(x, kmeans.labels_))
print(davies_bouldin_score(x, agg.labels_))
print(davies_bouldin_score(x, dbscan.labels_))

print(silhouette_score(x, kmeans.labels_))
print(silhouette_score(x, agg.labels_))
print(silhouette_score(x, dbscan.labels_))
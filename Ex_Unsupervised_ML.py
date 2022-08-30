# Unsupervised Machine Learning, solution by Patrik Svab

# Importing

import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# 1. Feature Engineering

# a) Loading

housing = fetch_california_housing()
x = housing["data"]

# b) Polynomial Features

poly = PolynomialFeatures(degree=2, include_bias=False)
new_x = poly.fit_transform(x)
new_names = poly.get_feature_names(housing["feature_names"])

print(poly.n_output_features_)
# There are 44 features now.

# c) Data Frame

df = pd.DataFrame(new_x, columns=new_names)
df["y"] = housing["target"]
df.to_csv("./output/polynomials.csv")

# 2. Principal Component Analysis

# a) Loading

FNAME = "./data/olympics.csv"
olymp = pd.read_csv(FNAME, sep=',', index_col=0)
olymp = olymp.drop(columns="score")

# We drop it because the overall score should be calculated from the
# individual disciplines as a linear combination.

# b) Scaling

scaler = StandardScaler()
scaler.fit(olymp)
olymp_scaled = pd.DataFrame(scaler.transform(olymp))

# Checking the unit variance:
print(olymp_scaled.var())

# c) PCA Method

pca = PCA(random_state=42)
pca.fit(olymp_scaled)
olymp_components = pd.DataFrame(pca.components_, columns=olymp.columns)

# Most prominent loading:
# On the first component, 100m sprint, 400m run,
#     and 110m hurdles load the best (and running long - in absolute values).
# On the second component, discus throw, "poid", and 1.500m run load the best.
# On the third component, "haut", 100m sprint, and javelin
#     load the best (and 1.500m run - in absolute values).
# The interpretation could be the following: the first component could reflect
#     speed, the second strength, and the third height
#     (or similar skills that attribute to the athlete's performance).

# d) Variation
df_var = pd.DataFrame(pca.explained_variance_ratio_)
df_var_cumul = df_var.cumsum()

# We need at least 7 components to explain 90% variation of the data.

# 3. Clustering

# a) Loading

iris = load_iris()
x2 = pd.DataFrame(iris["data"], columns=iris["feature_names"])

# b) Scaling

scaler = StandardScaler()
scaler.fit(x2)
x2_scaled = pd.DataFrame(scaler.transform(x2))
x2_scaled.var()

# c) Fitting Clusters

# K Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x2_scaled)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(x2_scaled)

# DBSCAN
dbscan = DBSCAN(eps=1, min_samples=2, metric='euclidean')
dbscan.fit(x2_scaled)

combined = pd.DataFrame({"kmeans": kmeans.labels_,
                         "agg": agg.labels_,
                         "dbscan": dbscan.labels_})

# d) Silhouette Scores

print(silhouette_score(x2_scaled, kmeans.labels_))
print(silhouette_score(x2_scaled, agg.labels_))
print(silhouette_score(x2_scaled, dbscan.labels_))

# DBSCAN has the highest silhouette score (0.50).

# Noise assignments from DBSCAN do not belong to any cluster (this method
# of clustering leaves some observations without any cluster).

# e) Adding Variables

combined["sepal width"] = x2["sepal width (cm)"]
combined["petal length"] = x2["petal length (cm)"]

# f) Renaming Noise Assignments

combined["dbscan"] = combined["dbscan"].replace(-1, "Noise")

# g) Plotting

id_vars = ["sepal width", "petal length"]
value_vars = ["kmeans", "agg", "dbscan"]
melted = combined.melt(id_vars=id_vars, value_vars=value_vars)
figure = sns.relplot(x="sepal width", y="petal length", data=melted,
                     hue="value", col="variable")
figure.savefig("./output/cluster_petal.pdf")

# It makes sense since the outlier is distant from both groups of observations.

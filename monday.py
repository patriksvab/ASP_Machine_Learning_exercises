# Errors

int("9.0")

marks=[1,1,4]
print(marks[4])

capitals = {'ger': 'berlin', 'aut': 'vienna'}
print(capitals['fra'])

my_list = "dbcea"
my_list.sort()


## Principal component analysis (= reducing the number of features)

import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


breastcancer = load_breast_cancer()
x = breastcancer["data"]
y = breastcancer["target"]

# Scale data
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

# Train model
pca = PCA(random_state=42)
pca.fit(x_scaled)

# Inspect components

pca.components_

sns.heatmap(pca.components_, xticklabels = breastcancer["feature_names"])

pca.explained_variance_ratio_

df = pd.DataFrame(pca.explained_variance_ratio_, columns=["Explained Variance"])

df["Cumulative"] = df["Explained Variance"].cumsum()

# We can take just the first six components and they will explain approx. 95% of the variance (we save 40 %)

df.plot(kind="bar")

data = pd.DataFrame(x, columns=breastcancer["feature_names"])
variances = data.describe()








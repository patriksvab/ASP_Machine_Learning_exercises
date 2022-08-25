# 1. Regularization

# a) Reading the Data

import matplotlib.pyplot
import pandas as pd
import seaborn as sns

FILENAME = "output/polynomials.csv"
regul = pd.read_csv(FILENAME, sep=',', index_col=0)

# b) Splitting Columns to X and y

y = regul["y"]
X = regul.drop(columns="y")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# c) OLS, Ridge, and Lasso

from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import Lasso, Ridge


lm = OLS().fit(X_train, y_train)
lm.score(X_test, y_test)

ridge = Ridge(alpha=0.3).fit(X_train, y_train)
ridge.score(X_test, y_test)

lasso = Lasso(alpha=0.3).fit(X_train, y_train)
lasso.score(X_test, y_test)

# OLS scores 0.656, Rigde scores 0.677, and Lasso scores 0.633, which makes Ridge the best model for predictions.

# d) Coefficients Comparison

comparison = pd.DataFrame(lm.coef_, index=X.columns, columns=["OLS"])
comparison["Lasso"] = lasso.coef_
comparison["Ridge"] = ridge.coef_

comparison[(comparison["Lasso"] == 0) & (comparison["Ridge"] != 0)]
sum((comparison["Lasso"] == 0) & (comparison["Ridge"] != 0))

# Both conditions are satisfied in 17 rows.

# e) Plotting

figure_comparison = comparison.plot.barh(figsize=(10, 30))
figure_comparison.figure.savefig("output/polynomials.pdf")


# 2. Neural Network Regression

# a) Loading and Splitting Data

import sklearn.datasets

diabetes = sklearn.datasets.load_diabetes()

X2 = diabetes["data"]
y2 = diabetes["target"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=42)

# b) Scaling, Neural Network Regressor

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# According to the Scikit Learn documentation: "For small datasets, ‘lbfgs’ can converge faster and perform better."

algorithms = [("scaler", StandardScaler()),
              ("nn", MLPRegressor(solver="lbfgs", random_state=42, max_iter=1000, activation="identity"))]
pipe = Pipeline(algorithms, verbose=True)
param_grid = {"nn__hidden_layer_sizes": [(60, 60), (100, 100), (140, 140)],
              "nn__alpha": [0.001, 0.01, 0.1]}
grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X2_train, y2_train)

# c) Best Parameters

results = pd.DataFrame(grid.cv_results_)
print(grid.best_estimator_)


# The model performs the best for alpha = 0.01 and hidden_layer_sizes = (100, 100).
# They reach a mean test score of 0.46724, which does not indicate a very good model.
# Missing something?

# d) Heatmap

best_model = grid.best_estimator_
bm = best_model._final_estimator
df2 = pd.DataFrame(bm.coefs_[0])
heatmap_diabetes = sns.heatmap(df2, yticklabels=diabetes["feature_names"])
heatmap_diabetes.figure.savefig("output/nn_diabetes_importances.pdf")

# 3. Neural Network Classification

# a) Loading Data

cancer = sklearn.datasets.load_breast_cancer()

X3 = cancer["data"]
y3 = cancer["target"]

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, random_state=42)

# b) Reading about ROC-AUC

# c) Neural Network Classifier

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

algorithms3 = [("scaler", MinMaxScaler()),
              ("nn", MLPClassifier(solver="lbfgs", random_state=42, max_iter=1000))]
pipe3 = Pipeline(algorithms3, verbose=True)
param_grid3 = {"nn__hidden_layer_sizes": [(100, 100), (150, 150)],
              "nn__alpha": [0.01, 0.1]}
grid3 = GridSearchCV(pipe3, param_grid3, scoring="roc_auc")
grid3.fit(X3_train, y3_train)

results_cancer = pd.DataFrame(grid3.cv_results_)
best_model_cancer = grid3.best_estimator_

# The model performs the best for alpha = 0.1 and hidden_layer_sizes = (150, 150).
# For these setting, the ROC-AuC-score is 0.993, which points to a very good performance of the model.

# d) Confusion Matrix as Heatmap

preds = grid3.predict(X3_test)
matrix = confusion_matrix(y3_test, preds)
heatmap_cancer = sns.heatmap(matrix, annot=True, xticklabels=cancer["target_names"],
                             yticklabels=cancer["target_names"])
heatmap_cancer.figure.savefig("./output/nn_breast_confusion.pdf")
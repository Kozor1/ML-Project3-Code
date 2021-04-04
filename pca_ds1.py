import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
import time
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, validation_curve
from sklearn import datasets as ds
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_curve, average_precision_score, adjusted_rand_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sn
from sklearn.neural_network import MLPClassifier

random_seed = 17

# ds = ds.make_classification(n_samples=2000, n_features=5, n_informative=3, n_clusters_per_class=3, random_state=random_seed)
# n=2; p=30

ds = ds.make_classification(n_samples=10000, n_features=10, n_informative=5, n_repeated=2,
                            n_clusters_per_class=5, flip_y=0.025, class_sep = 0.5, random_state=random_seed)
n=4; p=100

X, y = ds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

start_time = time.time()
pca = PCA(n_components=n)
pca_result = pca.fit_transform(X_train)
mid_time = time.time()

# # Defining Model
# model = TSNE(n_components=2, perplexity=p)
# # Fitting Model
# transformed = model.fit_transform(pca_result)
# end_time = time.time()

# # Plotting 2d t-Sne
# x_axis = transformed[:, 0]
# y_axis = transformed[:, 1]
#
# plt.scatter(x_axis, y_axis, c=y_train)
# plt.show()

opt_clf = MLPClassifier(activation='logistic', hidden_layer_sizes=[40,60], solver='lbfgs', alpha=0.1, random_state=random_seed)

X_train = pca_result
X_test = pca.fit_transform(X_test)

opt_clf.fit(X_train, y_train)
preds = opt_clf.predict(X_test)

accuracy_score = accuracy_score(preds, y_test)
end_time  = time.time()

print(mid_time - start_time)
print(end_time - mid_time)
print(accuracy_score)



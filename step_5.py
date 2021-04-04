import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
import time
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, validation_curve
from sklearn import datasets as ds
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_curve, average_precision_score, adjusted_rand_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap
from sklearn.random_projection import GaussianRandomProjection
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier

random_seed = 17

ds = ds.make_classification(n_samples=10000, n_features=10, n_informative=5, n_repeated=2,
                            n_clusters_per_class=5, flip_y=0.025, class_sep = 0.5, random_state=random_seed)
n=4; p=100

X, y = ds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Use one of the below for the training and test data for each method
# pca_result = PCA(n_components=n).fit_transform(X_train)
# lda_result = LinearDiscriminantAnalysis(n_components = n).fit_transform(X_train, y=y_train)
# rp_result = GaussianRandomProjection(n_components = n).fit_transform(X_train)
# iso_result = Isomap(n_components = n).fit_transform(X_train)

start_time = time.time()

# # K Means
# data_train = Isomap(n_components = n).fit_transform(X_train)
# clusterer = KMeans(n_clusters=15)
# cluster_labels = clusterer.fit_predict(data_train)
# cluster_labels = OneHotEncoder().fit_transform(cluster_labels.reshape(-1,1))
# X_train = scipy.sparse.hstack((X_train, cluster_labels))
#
# data_test = Isomap(n_components = n).fit_transform(X_test)
# clusterer = KMeans(n_clusters=15)
# cluster_labels = clusterer.fit_predict(data_test)
# cluster_labels = OneHotEncoder().fit_transform(cluster_labels.reshape(-1,1))
# X_test = scipy.sparse.hstack((X_test, cluster_labels))

# EM
data_train = LinearDiscriminantAnalysis(n_components = n).fit_transform(X_train, y=y_train)
clusterer = GaussianMixture(n_components=10)
cluster_labels = clusterer.fit_predict(data_train)
cluster_labels = OneHotEncoder().fit_transform(cluster_labels.reshape(-1,1))
X_train = scipy.sparse.hstack((X_train, cluster_labels))

data_test = LinearDiscriminantAnalysis(n_components = n).fit_transform(X_test, y=y_test)
clusterer = GaussianMixture(n_components=10)
cluster_labels = clusterer.fit_predict(data_test)
cluster_labels = OneHotEncoder().fit_transform(cluster_labels.reshape(-1,1))
X_test = scipy.sparse.hstack((X_test, cluster_labels))

opt_clf = MLPClassifier(activation='logistic', hidden_layer_sizes=[40,60], solver='lbfgs', alpha=0.1, random_state=random_seed)

opt_clf.fit(X_train, y_train)
preds = opt_clf.predict(X_test)

accuracy_score = accuracy_score(preds, y_test)
end_time  = time.time()

print(end_time - start_time)
print(accuracy_score)

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

def get_pc_info(X: pd.DataFrame, n_components: int=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ### BEGIN YOUR SOLUTION ###
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_
    return X_reduced, explained_variance_ratio, components
    ### END YOUR SOLUTION ###

def cluster_on_latent_space(
    X, reduction_model, y_true,
    n_clusters: int=7, random_state: int=42
) -> Tuple[float, float]:
    ### BEGIN YOUR SOLUTION ###
    if hasattr(reduction_model, 'fit_transform'):
        X_reduced = reduction_model.fit_transform(X)
    else:
        X_reduced = reduction_model.transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    y_pred = kmeans.fit_predict(X_reduced)
    ari = adjusted_rand_score(y_true, y_pred)
    sil = silhouette_score(X_reduced, y_pred)
    return ari, sil
    ### END YOUR SOLUTION ###
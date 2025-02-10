import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from logger import get_logger

logger = get_logger(__name__)

def perform_clustering(features_scaled_df):
    """Apply different clustering algorithms to the dataset."""
    logger.info("Executing function: perform_clustering")
    
    X_scaled = features_scaled_df.copy()
    
    hierarchical = AgglomerativeClustering(n_clusters=10, linkage='ward')
    X_scaled['hierarchical_cluster'] = hierarchical.fit_predict(features_scaled_df)
    
    gmm = GaussianMixture(n_components=10, random_state=42)
    X_scaled['gmm_cluster'] = gmm.fit_predict(features_scaled_df)
    
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    X_scaled['dbscan_cluster'] = dbscan.fit_predict(features_scaled_df)
    
    return X_scaled

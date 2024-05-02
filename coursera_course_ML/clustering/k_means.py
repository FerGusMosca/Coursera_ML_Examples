
from sklearn.cluster import KMeans as SkLearnKMeans

from coursera_course_ML.clustering.base_clustering_algo import BaseClusteringAlgo


class KMeans(BaseClusteringAlgo):
    def __init__(self):
        pass

    def do_cluster(self,x_norm,cluster_num,n_init):
        k_means = SkLearnKMeans(init="k-means++", n_clusters=cluster_num, n_init=n_init)
        k_means.fit(x_norm)
        labels = k_means.labels_
        return labels
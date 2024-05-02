import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from coursera_course_ML.clustering.k_means import KMeans
from coursera_course_ML.util.file_downloader import FileDownloader
from coursera_course_ML.util.plotter import Plotter


class KMeansTest():
    output_path = "./coursera_course_ML/input/k_means.csv"

    download_url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv"



    def __init__(self):
        pass


    @staticmethod
    def do_download():
        downloader = FileDownloader()
        downloader.download(KMeansTest.download_url, KMeansTest.output_path)
        k_means = KMeans()
        df = k_means.load_data(KMeansTest.output_path)
        return k_means, df

    @staticmethod
    def test():
        k_means, df = KMeansTest.do_download()

        df=k_means.drop_features(df,"Address")

        X = df.values[:, 1:]
        X = np.nan_to_num(X)

        X_norm=k_means.normalize(X)

        labels=k_means.do_cluster(X_norm,cluster_num=3,n_init=12)#labels is the cluster for every row

        df["Clus_km"] = labels#we create a new column for which we assign every label

        #we calculate th centroids
        centroids=df.groupby('Clus_km').mean()#for each cluster we can see the centroids

        #We draw a 2D graph of Age and income and the found clusters
        Plotter.plot_scatter_plot(X[:, 0],X[:, 3],area=np.pi * ( X_norm[:, 1])**2 ,labels=labels,
                                  X_legend="Age",Y_legend="Income")

        #Then a 3D with 3 variables
        Plotter.scatter_3D(X[:, 1], X[:, 0], X[:, 3], labels=labels, X_legend="Education", Y_legend="Age",Z_legend="Income")

        print("aca")



    @staticmethod
    def draw_random_data():
        np.random.seed(0)
        X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
        plt.scatter(X[:, 0], X[:, 1], marker='.')
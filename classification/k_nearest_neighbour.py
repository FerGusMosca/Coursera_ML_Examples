import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

from classification.base_classification_algo import BaseClasifficationAlgo


class KNearestNeighbour(BaseClasifficationAlgo):


    #region Constructors

    def __init__(self):
        pass

    #endregion

    #region Private Methods




    def normalize(self,X_train):
        X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
        return X_train_norm

    def train_model(self,X_train_norm,y_train,k):
        # Train Model and Predict
        neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train_norm, y_train)
        return neigh

    def predict(self,neigh,X_test_norm):
        yhat = neigh.predict(X_test_norm)
        return  yhat



    def plot_k_vs_acc(self,Ks,mean_acc,std_acc):
        plt.plot(range(1, Ks), mean_acc, 'g')
        plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
        plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
        plt.legend(('Accuracy ', '+/- 1xstd', '+/- 3xstd'))
        plt.ylabel('Accuracy ')
        plt.xlabel('Number of Neighbors (K)')
        plt.tight_layout()
        plt.show()


    #endregion






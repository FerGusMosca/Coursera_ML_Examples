
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import requests
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

class KNearestNeighbour():

    path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"

    #region Constructors

    def __init__(self):
        pass

    #endregion

    #region Private Methods

    def load_data(self, path):
        df = pd.read_csv(path)#panda datasets
        df.head()
        print("Succesfully loaded {}".format(path))
        return df

    def get_X_axis(self, df, indep_cols):
        # .astype(float)
        X = df[indep_cols].values
        return X

    def get_Y_axis(self, df, dep_cols):
        Y = df[dep_cols].values
        return Y

    def split_data(self,X,Y):
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=4)
        return (X_train, X_test, y_train, y_test)

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

    def calculate_accuracy(self,y_test,y_pred):
        return metrics.accuracy_score(y_test, y_pred)

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






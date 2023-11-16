import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

class BaseClasifficationAlgo():

    def __init__(self):
        pass


    #region Public Methods

    def load_data(self, path):
        df = pd.read_csv(path)#panda datasets
        df.head()
        print("Succesfully loaded {}".format(path))
        return df

    #Extracts all the columns from the df array
    def get_X_axis(self, df, indep_cols):
        X = df[indep_cols].values
        return X
    #Same w/Y axis
    def get_Y_axis(self, df, dep_cols):
        Y = df[dep_cols].values
        return Y

    #replaces all the categorical values  (ex: Male, Female) w/numerical values
    def do_transform(self,X,values,col):
        le_trasf = preprocessing.LabelEncoder()
        le_trasf.fit(values)
        X[:, col] = le_trasf.transform(X[:, col])

    def split_data(self,X,Y,test_size=0.2,random_state=4):
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=test_size, random_state=random_state)
        return (X_train, X_test, y_train, y_test)

    def calculate_accuracy(self,y_test,y_pred):
        return metrics.accuracy_score(y_test, y_pred)


    def jaccard_index(self,y_test,y_pred,pos_label=0):
        return jaccard_score(y_test, y_pred, pos_label=pos_label)

    def get_confussion_matrix(self,y_test,y_pred,labels=[1,0],precision=2):
        cnf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
        np.set_printoptions(precision=precision)

        # Plot non-normalized confusion matrix
        return  cnf_matrix

    def log_loss(self,y_test,y_pred):
        #Weakness of the probabilistic estimation
        #Average of allt he probabilistic estimations weakness
        #Ex; a prob of 0.6, is 0.6 from 0 and 0.4 away from 1.--> Weak prediction
        return log_loss(y_test, y_pred)


    #endregion
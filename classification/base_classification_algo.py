import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

from Common.BaseAlgo import BaseAlgo


class BaseClasifficationAlgo(BaseAlgo):

    def __init__(self):
        pass


    #region Public Methods



    #Extracts all the columns from the df array
    def get_X_axis(self, df, indep_cols):
        X = df[indep_cols].values
        return X

    def get_X_axis_as_arr(self,cell_df,cols):
        feature_df = cell_df[cols]
        X = np.asarray(feature_df)
        return X

    #Same w/Y axis
    def get_Y_axis(self, df, dep_cols):
        Y = df[dep_cols].values
        return Y

    def get_Y_axis_as_arr(self, cell_df, col,type):
        cell_df[col] = cell_df[col].astype(type)
        y = np.asarray(cell_df[col])
        return y

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

    def f1_score_index(self,y_test,y_pred,averages='weighted'):
        return f1_score(y_test, y_pred, average=averages)

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

    #removes all the ros whose "col" is not of type exp_type
    def clean_dataset(self,cell_df,col, exp_type):
        #pd.to_numeric(cell_df[col], errors='coerce') --> All the values to nnumeric. Otherwise--> Nan
        # values.notnull() --> If not null, True, else, False ==> All the non numeric values will be False
        cell_df = cell_df[pd.to_numeric(cell_df[col], errors='coerce').notnull()]

        #All the remaining values will be converted as exp_type
        cell_df[col] = cell_df[col].astype(exp_type)#Converts all the values in "col" to "exp_type"
        return  cell_df


    #endregion
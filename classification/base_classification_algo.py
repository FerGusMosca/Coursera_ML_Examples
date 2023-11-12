import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split


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


    #endregion
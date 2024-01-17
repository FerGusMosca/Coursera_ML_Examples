from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
from framework.common.logger.message_type import MessageType
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle

_OUTPUT_PATH="./output/"
_LOGISTIC_REGRESSION_MODEL_NAME="logistic_regression"
_SVM_MODEL_NAME="support_vector_machine"
_KNN_MODEL_NAME="k_nearest_neighbour"
_DECISSION_TREE_MODEL_NAME="decission_tree"

class MLModelAnalyzer():


    def __init__(self,p_logger):
        self.logger=p_logger


    def persist_model(self,trained_algo,algo_name):
        file_path="{}{}".format(_OUTPUT_PATH,"{}.pkl".format(algo_name))
        with open(file_path, 'wb') as file:
            pickle.dump(trained_algo, file)

    def fetch_model(self,algo_name):
        file_path = "{}{}".format(_OUTPUT_PATH, "{}.pkl".format(algo_name))
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
            return loaded_model


    def get_int_mapping(self,df,col):
        unique_values = df[col].unique()
        i=0
        mapping={}
        for val in unique_values:
            mapping[val]=i
            i+=1

        return mapping

    def map_categorical_Y(self,df_Y,classification_col):
        # We normalize the Y categorical axis
        mapping = self.get_int_mapping(df_Y, classification_col)  # We gte SHORT=1,LONG=2, etc.
        Y = df_Y[classification_col].map(mapping)
        return Y

    def normalize_X(self,df_X):
        # We need to normalize just the numeric columns of the X dataframe
        X_num_cols = df_X.select_dtypes(include='number').columns
        X_non_num_cols = df_X.columns.difference(X_num_cols)

        X_numeric = df_X[X_num_cols]
        X_num_scal = preprocessing.StandardScaler().fit_transform(X_numeric)
        X = pd.concat([pd.DataFrame(X_num_scal, columns=X_num_cols), df_X[X_non_num_cols]], axis=1)
        # After the concat --> THe X will have all its numeric column values properly fit
        return X

    def extract_non_numeric(self,X):
        X_num_cols = X.select_dtypes(include='number').columns
        X_numeric = X[X_num_cols]
        return  X_numeric

    def run_logistic_regression_eval(self,X_train, y_train,X_test,y_test):
        # TRAINING - Logistic Regression w/GridSearchcv
        resp_row = {'Model': 'Logistic Regression', 'Train Accuracy': None,'Test Accuracy':None}

        X_train_num=self.extract_non_numeric(X_train)

        parameters = {"C": [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
        lr = LogisticRegression()
        logreg_cv = GridSearchCV(lr, parameters)
        logreg_cv.fit(X_train_num, y_train)
        self.logger.do_log("tuned hpyerparameters :(best parameters):{} ".format( logreg_cv.best_params_),MessageType.INFO)
        self.logger.do_log("Logistic Regression - Training Params Accuracy :{}".format(logreg_cv.best_score_), MessageType.INFO)
        resp_row["Train Accuracy"] = logreg_cv.best_score_

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.extract_non_numeric(X_test)
        lr_accuracy = logreg_cv.score(X_test_num, y_test)
        resp_row["Test Accuracy"] = lr_accuracy
        self.logger.do_log("Logistic Regression - Test Accuracy Score :{}".format(lr_accuracy),MessageType.INFO)


        self.persist_model(logreg_cv.best_estimator_,_LOGISTIC_REGRESSION_MODEL_NAME)

        #yhat = logreg_cv.predict(X_test)
        # self.plot_confusion_matrix(y_test, yhat)

        return  resp_row

    def build_out_of_sample_report_row(self,name,y_test,y_hat):
        resp_row = {'Model': name,'Accuracy':None,'Precision':None,'Recall':None,'F1':None}

        accuracy = accuracy_score(y_test, y_hat)
        precision = precision_score(y_test, y_hat)
        recall = recall_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat)
        cm = confusion_matrix(y_test, y_hat)
        report = classification_report(y_test, y_hat)

        resp_row["Accuracy"]=accuracy
        resp_row["Precision"] = precision
        resp_row["F1"] = f1
        resp_row["Recall"] = recall
        return resp_row


    def run_logistic_regression_eval_out_of_sample(self,X_test,y_test):
        # TRAINING - Logistic Regression w/GridSearchcv
        lr = self.fetch_model(_LOGISTIC_REGRESSION_MODEL_NAME)

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.extract_non_numeric(X_test)
        y_hat = lr.predict(X_test_num)
        return  self.build_out_of_sample_report_row("Logistic Regression",y_test,y_hat)

    def run_support_vector_machine_eval_out_of_sample(self,X_test,y_test):
        # TRAINING - SVM w/GridSearchcv
        svm = self.fetch_model(_SVM_MODEL_NAME)

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.extract_non_numeric(X_test)
        y_hat = svm.predict(X_test_num)
        return  self.build_out_of_sample_report_row("Support Vector Machine",y_test,y_hat)

    def run_decission_tree_eval_out_of_sample(self,X_test,y_test):
        # TRAINING - Decission Tree w/GridSearchcv
        dec_tree = self.fetch_model(_DECISSION_TREE_MODEL_NAME)

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.extract_non_numeric(X_test)
        y_hat = dec_tree.predict(X_test_num)
        return  self.build_out_of_sample_report_row("Decission Tree",y_test,y_hat)

    def run_K_nearest_neighbour_eval_out_of_sample(self,X_test,y_test):
        # TRAINING - K-Nearest Neighbour w/GridSearchcv
        knn = self.fetch_model(_KNN_MODEL_NAME)

        # TEST - Accuracy of test data versus predictions + conf. matrix of predictions
        X_test_num = self.extract_non_numeric(X_test)
        y_hat = knn.predict(X_test_num)
        return  self.build_out_of_sample_report_row("K-Nearest Neighbour",y_test,y_hat)

    def run_support_vector_machine_eval(self, X_train, y_train, X_test, y_test):
        # TRAINING - SVM w/GridSearchcv
        resp_row = {'Model': 'Support Vector Machine', 'Train Accuracy': None, 'Test Accuracy': None}

        X_train_num = self.extract_non_numeric(X_train)

        parameters = {'kernel': ('linear', 'rbf', 'poly', 'rbf', 'sigmoid'),
                      'C': np.logspace(-3, 3, 5),
                      'gamma': np.logspace(-3, 3, 5)}
        svm = SVC()
        svm_cv = GridSearchCV(svm, parameters)
        svm_cv.fit(X_train_num, y_train)
        self.logger.do_log("tuned hpyerparameters :(best parameters):{} ".format( svm_cv.best_params_),MessageType.INFO)
        self.logger.do_log("Support Vector Machine - Training Params Accuracy :{}".format( svm_cv.best_score_),MessageType.INFO)
        resp_row["Train Accuracy"] =svm_cv.best_score_

        # TEST Calculate the accuracy on the test data using the method score
        X_test_num = self.extract_non_numeric(X_test)
        svm_accuracy = svm_cv.score(X_test_num, y_test)
        resp_row["Test Accuracy"] = svm_accuracy
        self.logger.do_log("Support Vector Machine - Test Accuracy Score :{}".format(svm_accuracy), MessageType.INFO)

        #yhat = logreg_cv.predict(X_test)
        #self.plot_confusion_matrix(y_test, yhat)
        self.persist_model(svm_cv.best_estimator_, _SVM_MODEL_NAME)

        return resp_row

    def run_decision_tree_eval(self, X_train, y_train, X_test, y_test,reuse_last=False):
        # TRAINING - Decission Tree w/GridSearchcv
        resp_row = {'Model': 'Decision Tree', 'Train Accuracy': None, 'Test Accuracy': None}

        X_train_num = self.extract_non_numeric(X_train)

        parameters = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_depth': [2 * n for n in range(1, 10)],
                      'max_features': ['auto', 'sqrt'],
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10]}

        tree = DecisionTreeClassifier()
        tree_cv = GridSearchCV(tree, parameters)
        tree_cv.fit(X_train_num, y_train)
        self.logger.do_log("tuned hpyerparameters :(best parameters):{} ".format( tree_cv.best_params_),MessageType.INFO)
        self.logger.do_log("Decision Tree - Training Params Accuracy :{}".format(tree_cv.best_score_),MessageType.INFO)
        resp_row["Train Accuracy"] = tree_cv.best_score_

        # TEST Calculate the accuracy on the test data using the method score
        X_test_num = self.extract_non_numeric(X_test)
        tree_accuracy = tree_cv.score(X_test_num, y_test)
        self.logger.do_log("Decision Tree - Test Accuracy Score :{}".format(tree_accuracy),MessageType.INFO)
        resp_row["Test Accuracy"] = tree_accuracy

        self.persist_model(tree_cv.best_estimator_, _DECISSION_TREE_MODEL_NAME)

        return resp_row

    def run_k_nearest_neighbour_eval(self, X_train, y_train, X_test, y_test):
        # TRAINING - KNN w/GridSearchcv
        resp_row = {'Model': 'K Nearest Neighbour', 'Train Accuracy': None, 'Test Accuracy': None}

        X_train_num = self.extract_non_numeric(X_train)

        parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'p': [1, 2]}

        KNN = KNeighborsClassifier()
        knn_cv = GridSearchCV(KNN, parameters)
        knn_cv.fit(X_train_num, y_train)
        self.logger.do_log("tuned hpyerparameters :(best parameters):{} ".format( knn_cv.best_params_),MessageType.INFO)
        self.logger.do_log("KNN - Training Params Accuracy :{}".format( knn_cv.best_score_),MessageType.INFO)
        resp_row["Train Accuracy"] = knn_cv.best_score_

        # TEST Calculate the accuracy on the test data using the method score
        X_test_num = self.extract_non_numeric(X_test)
        knn_accuracy = knn_cv.score(X_test_num, y_test)
        self.logger.do_log("KNN - Test Accuracy Score :{}".format(knn_accuracy),MessageType.INFO)
        resp_row["Test Accuracy"] = knn_accuracy

        self.persist_model(knn_cv.best_estimator_, _KNN_MODEL_NAME)
        return resp_row


    def fit_and_evaluate(self,series_df,classification_col):
        comparisson_df = pd.DataFrame(columns=['Model','Train Accuracy','Test Accuracy'])

        # STEP 1 - Split the dataframe inot X (indep. variable) and Y (dep variable)
        features=series_df.columns.to_list()
        features.remove(classification_col)
        df_X = series_df[features]
        df_Y = series_df[[classification_col]]

        #STEP 2 - Transform categorical values domension + Normalize the data
        df_X = pd.get_dummies(df_X)  # converts the categorical values into mult. cols

        #We map the Y categorical axis to int values
        Y=self.map_categorical_Y(df_Y, classification_col)

        #Then we normalize all the numerical values of X
        X=self.normalize_X(df_X)

        #STEP 3 - Split the data into training and test
        X_train, X_test, y_train, y_test =train_test_split(X, Y,test_size=0.2,random_state=2)

        #LOGISTIC REGRESSION
        resp_row= self.run_logistic_regression_eval(X_train, y_train,X_test,y_test)
        comparisson_df=comparisson_df.append(resp_row, ignore_index=True)

        # SUPPORT VECTOR MACHINE
        resp_row = self.run_support_vector_machine_eval(X_train, y_train, X_test, y_test)
        comparisson_df = comparisson_df.append(resp_row, ignore_index=True)

        # DECISSION TREE
        resp_row = self.run_decision_tree_eval(X_train, y_train, X_test, y_test)
        comparisson_df = comparisson_df.append(resp_row, ignore_index=True)

        # K NEAREST NEIGHBOUR
        resp_row = self.run_k_nearest_neighbour_eval(X_train, y_train, X_test, y_test)
        comparisson_df = comparisson_df.append(resp_row, ignore_index=True)

        return comparisson_df

    def fetch_and_evaluate(self,series_df,classification_col):
        comparisson_df = pd.DataFrame(columns=['Model','Accuracy','Precision','Recall','F1'])

        # STEP 1 - Split the dataframe inot X (indep. variable) and Y (dep variable)
        features=series_df.columns.to_list()
        features.remove(classification_col)
        df_X = series_df[features]
        df_Y = series_df[[classification_col]]

        #STEP 2 - Transform categorical values domension + Normalize the data
        df_X = pd.get_dummies(df_X)  # converts the categorical values into mult. cols

        #We map the Y categorical axis to int values
        Y=self.map_categorical_Y(df_Y, classification_col)

        #Then we normalize all the numerical values of X
        X=self.normalize_X(df_X)

        #LOGISTIC REGRESSION
        resp_row= self.run_logistic_regression_eval_out_of_sample(X,Y)
        comparisson_df=comparisson_df.append(resp_row, ignore_index=True)

        # # SUPPORT VECTOR MACHINE
        resp_row = self.run_support_vector_machine_eval_out_of_sample(X,Y)
        comparisson_df = comparisson_df.append(resp_row, ignore_index=True)

        # # DECISSION TREE
        resp_row = self.run_decission_tree_eval_out_of_sample(X,Y)
        comparisson_df = comparisson_df.append(resp_row, ignore_index=True)

        # # K NEAREST NEIGHBOUR
        resp_row = self.run_K_nearest_neighbour_eval_out_of_sample(X,Y)
        comparisson_df = comparisson_df.append(resp_row, ignore_index=True)

        return comparisson_df
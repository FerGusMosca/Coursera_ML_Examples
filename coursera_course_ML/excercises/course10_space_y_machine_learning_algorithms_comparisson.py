# Pandas is a software library written for the Python programming language for data manipulation and analysis.
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier

import plotly.express as px
import pandas as pd

from coursera_course_ML.exams.data_science_IBM_cert.exams.exam10_presentation_calculations import \
    Exam10PresentationCalculations


class Course10SpaceYMachineLearningPrediction:
    _CSV_PATH = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv'

    _CSV_PATH_2="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv"

    def __init__(self):
        pass

    def plot_confusion_matrix(self,y, y_predict):
        "this function plots the confusion matrix"
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y, y_predict)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax);  # annot=True to annotate cells
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix');
        ax.xaxis.set_ticklabels(['did not land', 'land']);
        ax.yaxis.set_ticklabels(['did not land', 'landed'])
        plt.show()

    def visualize_model_comparisson(self,comparisson_df):
        sns.barplot(x='Model', y='Accuracy', data=comparisson_df)

        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of the different models')

        plt.show()


    def compare_ML_algorithms(self):
        df_1 = pd.read_csv(Course10SpaceYMachineLearningPrediction._CSV_PATH)

        comparisson_df = pd.DataFrame(columns=['Model', 'Accuracy'])

        #df_2 = pd.read_csv(Course10SpaceYMachineLearningPrediction._CSV_PATH_2)

        #TASK 1 - Split the dataframe inot X (indep. variable) and Y (dep variable)
        features=['BoosterVersion', 'PayloadMass', 'Orbit',
                   'LaunchSite', 'Outcome', 'Flights', 'GridFins', 'Reused', 'Legs',
                   'LandingPad', 'Block', 'ReusedCount', 'Serial', 'Longitude', 'Latitude']

        df_X = df_1[features]
        df_Y = df_1[["Class"]]

        #TASK 2 - Transform categorical values domension + Normalize the data
        df_X = pd.get_dummies(df_X)  # converts the categorical values into mult. cols
        X = np.asarray(df_X)
        Y = np.asarray(df_Y.astype("int"))
        X = preprocessing.StandardScaler().fit_transform(X)

        #TASK 3 - Split the data into training and test
        X_train, X_test, y_train, y_test =train_test_split(X, Y,test_size=0.2,random_state=2)

        #TASK 4 - Logistic Regression w/GridSearchcv
        parameters = {"C": [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
        lr=LogisticRegression()
        logreg_cv = GridSearchCV(lr, parameters)
        logreg_cv.fit(X_train,y_train)
        print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
        print("Logistic Regression - Training Params Accuracy :", logreg_cv.best_score_)

        new_row = {'Model': 'Logistic Regression', 'Accuracy': logreg_cv.best_score_}
        comparisson_df = comparisson_df.append(new_row, ignore_index=True)

        #TASK 5 - Accuracy of test data versus predictions + conf. matrix of predictions
        yhat = logreg_cv.predict(X_test)
        #self.plot_confusion_matrix(y_test, yhat)
        lr_accuracy = logreg_cv.score(X_test, y_test)
        print("Logistic Regression - Test Accuracy Score :{}".format(lr_accuracy))

        #TASK6 -Support Vector Machine w/GridSearchCV
        parameters = {'kernel': ('linear', 'rbf', 'poly', 'rbf', 'sigmoid'),
                      'C': np.logspace(-3, 3, 5),
                      'gamma': np.logspace(-3, 3, 5)}
        svm = SVC()
        svm_cv = GridSearchCV(svm , parameters)
        svm_cv.fit(X_train, y_train)
        print("tuned hpyerparameters :(best parameters) ", svm_cv.best_params_)
        print("Support Vector Machine - Training Params Accuracy :", svm_cv.best_score_)
        new_row = {'Model': 'Support Vector Machine', 'Accuracy': svm_cv.best_score_}
        comparisson_df = comparisson_df.append(new_row, ignore_index=True)

        #TASK 7- Calculate the accuracy on the test data using the method score
        svm_accuracy = svm_cv.score(X_test, y_test)
        print("Support Vector Machine - Test Accuracy Score :{}".format(svm_accuracy))
        #print("accuracy :", svm_cv.best_score_)
        yhat = logreg_cv.predict(X_test)
        self.plot_confusion_matrix(y_test, yhat)


        #TASK 8 - Decission Tree Classifier
        parameters = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_depth': [2 * n for n in range(1, 10)],
                      'max_features': ['auto', 'sqrt'],
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10]}

        tree = DecisionTreeClassifier()
        tree_cv = GridSearchCV(tree, parameters)
        tree_cv.fit(X_train, y_train)
        print("tuned hpyerparameters :(best parameters) ", tree_cv.best_params_)
        print("Decision Tree - Training Params Accuracy :", tree_cv.best_score_)
        tree_accuracy = tree_cv.score(X_test, y_test)
        print("Decision Tree - Test Accuracy Score :{}".format(tree_accuracy))
        new_row = {'Model': 'Decision Tree', 'Accuracy': tree_cv.best_score_}
        comparisson_df = comparisson_df.append(new_row, ignore_index=True)

        #TASK 10 - K-nearest Neighbour
        parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'p': [1, 2]}

        KNN = KNeighborsClassifier()
        knn_cv = GridSearchCV(KNN, parameters)
        knn_cv.fit(X_train, y_train)
        print("tuned hpyerparameters :(best parameters) ", knn_cv.best_params_)
        print("KNN - Training Params Accuracy :", knn_cv.best_score_)
        knn_accuracy = knn_cv.score(X_test, y_test)
        print("KNN - Test Accuracy Score :{}".format(knn_accuracy))
        new_row = {'Model': 'K Nearest Neighbour', 'Accuracy': knn_cv.best_score_}
        comparisson_df = comparisson_df.append(new_row, ignore_index=True)

        self.visualize_model_comparisson(comparisson_df)
        pass


    def get_summmary_df_w_grid_fins_legss(self, launch_site=None, orbit=None):
        df_1 = pd.read_csv("{}space_x_launches.csv".format(Exam10PresentationCalculations._INPUT_FOLDER))

        if launch_site is not None:
            df_1=df_1[df_1["LaunchSite"]==True]

        if orbit is not None:
            df_1=df_1[df_1["Orbit"]==True]

        df_1 = df_1[df_1["GridFins"] == True]
        df_1 = df_1[df_1["Legs"] == True]



        success_df = df_1[(df_1["Outcome"] == "True Ocean")
                          | (df_1["Outcome"] == "True RTLS")
                          | (df_1["Outcome"] == "True ASDS")]
        success_df["Status"] = "Success"

        failed_df = df_1[(df_1["Outcome"] != "True Ocean")
                         & (df_1["Outcome"] != "True RTLS")
                         & (df_1["Outcome"] != "True ASDS")]
        failed_df["Status"] = "Failure"

        summary_df = pd.concat([success_df, failed_df])

        return summary_df

    def custom_calculations(self):

        #client=LaunchesServiceClient()
        #client.get_all_launches()

        summary_df=self.get_summmary_df_w_grid_fins_legss()
        #summary_df=summary_df.groupby("Status").count().reset_index().rename(columns={'FlightNumber': 'Count'})

        #1- Show the real success rate
        fig = px.pie(summary_df, names='Status', title='Mission Status Distribution',
                     labels={'Status': 'Mission Status'},
                     color_discrete_map={'Success': 'green', 'Failure': 'red'})

        # Show plot
        fig.show()

        #2- Scatter bettwen Payload Mass and Orbit
        fig = px.scatter(summary_df, x='PayloadMass', y='Orbit', color='Status',
                         title='Payload Mass vs Orbit with Status',
                         labels={'PayloadMass': 'Payload Mass', 'Orbit': 'Orbit Type'},
                         color_discrete_map={'Success': 'blue', 'Failure': 'red'})

        # Show plot
        fig.show()

        # 3- Scatter bettwen Payload Mass and LaunchSite
        fig = px.scatter(summary_df, x='PayloadMass', y='LaunchSite', color='Status',
                         title='Payload Mass vs LaunchSite with Status',
                         labels={'PayloadMass': 'Payload Mass', 'LaunchSite': 'LaunchSite'},
                         color_discrete_map={'Success': 'blue', 'Failure': 'red'})

        # Show plot
        fig.show()

        pass
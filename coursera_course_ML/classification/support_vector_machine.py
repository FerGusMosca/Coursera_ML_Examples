
import matplotlib.pyplot as plt
from sklearn import svm

from coursera_course_ML.classification.base_classification_algo import BaseClasifficationAlgo


class SupportVectorMachine(BaseClasifficationAlgo):

    def __init__(self):
        pass


    def plot_classification(self,cell_df,feature_1,feature_2,clasif_col,cat_1_val,cat_2_val,
                            cat_1_colog='DarkBlue',cat_2_color='Red', max_samples =50):

        ax = cell_df[cell_df[clasif_col] == 4][0:max_samples].plot(kind='scatter', x=feature_1, y=feature_2, color=cat_1_colog,
                                                                   label=cat_1_val);
        cell_df[cell_df[clasif_col] == 2][0:max_samples].plot(kind='scatter', x=feature_1, y=feature_2, color=cat_2_color,
                                                              label=cat_2_val, ax=ax);
        plt.show()

    def train_model(self,X_train,y_train,kernel="rbf"):
        clf = svm.SVC(kernel=kernel)
        clf.fit(X_train, y_train)
        return clf

    def predict(self,clf,X_test):
        yhat = clf.predict(X_test)
        return  yhat
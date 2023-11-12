from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from classification.base_classification_algo import BaseClasifficationAlgo


class DecissionTree(BaseClasifficationAlgo):

    def __init__(self):
        pass

    #region Public Methods

    def train_model(self,X_train, y_train,criterion="entropy",max_depth=4):
        drugTree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        drugTree.fit(X_train, y_train)
        return drugTree

    def predict(self,tree, X_test):
        predTree = tree.predict(X_test)
        return predTree

    def plot(self,drugTree):
        tree.plot_tree(drugTree)
        plt.show()


    #endregion
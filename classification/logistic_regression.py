from classification.base_classification_algo import BaseClasifficationAlgo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SkLearnLogRegression
from sklearn.metrics import confusion_matrix

class LogisticRegression(BaseClasifficationAlgo):

    def __init__(self):
        pass

    def train_model(self,X_train,y_train,c=0.01,solver="liblinear"):
        #C= inverse of regularization strength--> technique used to solve overfitting <smaller= stronger reg>
        LR = SkLearnLogRegression(C=c, solver=solver).fit(X_train, y_train)
        return  LR #LR is the model

    def predict(self,LR,X_test):#LR is the model trained
        yhat = LR.predict(X_test)
        return  yhat

    def predict_probabilities(self,LR,X_test):#LR is the model trained
        yhat = LR.predict_proba(X_test)
        return  yhat
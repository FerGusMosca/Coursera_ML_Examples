from coursera_course_ML.classification.logistic_regression import LogisticRegression
from coursera_course_ML.util.file_downloader import FileDownloader
from coursera_course_ML.util.plotter import Plotter


class LogisticRegressionTest():
    output_path = "./coursera_course_ML/input/logistic_regression.csv"

    download_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"

    def __init__(self):
        pass

    @staticmethod
    def do_download():
        downloader = FileDownloader()
        downloader.download(LogisticRegressionTest.download_url, LogisticRegressionTest.output_path)
        logistic_regr = LogisticRegression()
        df = logistic_regr.load_data(LogisticRegressionTest.output_path)
        return logistic_regr, df

    @staticmethod
    def split_datasets(logistic_regr,df):
        X = logistic_regr.get_X_axis(df,
                                 ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn'])

        Y = logistic_regr.get_Y_axis(df, 'churn').astype('int')

        # we split the dataset
        X_train, X_test, y_train, y_test = logistic_regr.split_data(X, Y)

        # no need to normalize the dataset
        return X_train, X_test, y_train, y_test

    @staticmethod
    def train_and_predict(logistic_regr,X_train,y_train,X_test,y_test):
        # we train the model
        model = logistic_regr.train_model(X_train, y_train)

        # we run some predictions w/out of sample arrays
        y_pred = logistic_regr.predict(model, X_test)

        # we run some predictions w/out of sample arrays
        y_prob = logistic_regr.predict_probabilities(model, X_test)

        return y_pred,y_prob


    @staticmethod
    def test():
        logistic_regr,df = LogisticRegressionTest.do_download()
        X_train, X_test, y_train, y_test=LogisticRegressionTest.split_datasets(logistic_regr,df)
        y_pred,y_prob=LogisticRegressionTest.train_and_predict(logistic_regr,X_train, y_train,X_test , y_test)


        print("Prediction successfully finished...")
        print("Jaccard index:{}".format(logistic_regr.jaccard_index(y_test,y_pred)))
        print("Log Loss:{}".format(logistic_regr.log_loss(y_test, y_prob)))

        conf_matrix=logistic_regr.get_confussion_matrix(y_test,y_pred)

        Plotter.plot_confussion_matrix(conf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')



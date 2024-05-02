from random import randint

from coursera_course_ML.classification.decission_tree import DecissionTree
from coursera_course_ML.util.file_downloader import FileDownloader


class DecissionTreeTest:
    output_path = "./coursera_course_ML/input/decission_tree_drug.csv"

    download_url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"

    def __init__(self):
        pass


    @staticmethod
    def do_download():
        downloader = FileDownloader()
        downloader.download(DecissionTreeTest.download_url, DecissionTreeTest.output_path)
        decission_tree_algo = DecissionTree()
        df = decission_tree_algo.load_data(DecissionTreeTest.output_path)
        return decission_tree_algo, df

    @staticmethod
    def split_datasets(dec_tree_algo,df):
        X = dec_tree_algo.get_X_axis(df,['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])

        Y = dec_tree_algo.get_Y_axis(df, 'Drug')

        dec_tree_algo.do_transform(X,['F','M'],1)
        dec_tree_algo.do_transform(X, [ 'LOW', 'NORMAL', 'HIGH'], 2)
        dec_tree_algo.do_transform(X, ['NORMAL', 'HIGH'], 3)


        # # we split the dataset
        X_train, X_test, y_train, y_test = dec_tree_algo.split_data(X, Y,test_size=0.3,random_state=3)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def train_and_predict(dec_tree_algo,X_train,y_train,X_test,y_test):
        # we train the model
        tree = dec_tree_algo.train_model(X_train, y_train)

        # # we run some predictions w/out of sample arrays
        y_pred = dec_tree_algo.predict(tree, X_test)

        print("==== Example for predicting a random inpput ====")
        index=randint(0,len(y_pred))
        print("random index {} for X:{}".format(index,X_test[index]))
        print("pred Y:{}".format(y_pred[index]))

        # # Evalute the prediction accuracy
        accuracy = dec_tree_algo.calculate_accuracy(y_test, y_pred)
        print("Algo accuracy:{0:.2f}%".format(accuracy*100))
        #
        return tree,y_pred,accuracy


    #region Public Methods
    @staticmethod

    def test_single_prediction(K=4):

        decission_tree_algo,df=DecissionTreeTest.do_download()

        X_train, X_test, y_train, y_test =DecissionTreeTest.split_datasets(decission_tree_algo,df)

        tree,y_pred,accuracy=DecissionTreeTest.train_and_predict(decission_tree_algo,X_train,y_train,X_test,y_test)

        decission_tree_algo.plot(tree)






    #endregion

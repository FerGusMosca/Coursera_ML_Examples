import numpy as np

from classification.k_nearest_neighbour import KNearestNeighbour
from util.file_downloader import FileDownloader


class KNearestTest():
    output_path = "./input/k-neirest-neightbour.csv"

    download_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"

    def __init__(self):
        pass

    @staticmethod
    def do_download():
        downloader = FileDownloader()
        downloader.download(KNearestTest.download_url, KNearestTest.output_path)
        k_nearest = KNearestNeighbour()
        df = k_nearest.load_data(KNearestTest.output_path)
        return k_nearest,df

    @staticmethod
    def split_datasets(k_nearest,df):
        X = k_nearest.get_X_axis(df,
                                 ['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire',
                                  'gender', 'reside'])

        Y = k_nearest.get_Y_axis(df, 'custcat')

        # we split the dataset
        X_train, X_test, y_train, y_test = k_nearest.split_data(X, Y)

        # we normalize the dataset
        X_train_norm = k_nearest.normalize(X_train)

        X_test_norm = k_nearest.normalize(X_test)

        return X_train, X_test, y_train, y_test,X_train_norm,X_test_norm

    @staticmethod
    def train_and_predict(k_nearest,X_train_norm,y_train,X_test_norm,y_test,k):
        # we train the model
        model = k_nearest.train_model(X_train_norm, y_train, k=k)

        # we run some predictions w/out of sample arrays
        y_pred = k_nearest.predict(model, X_test_norm)

        # Evalute the prediction accuracy
        accuracy = k_nearest.calculate_accuracy(y_test, y_pred)

        return model,y_pred,accuracy

    @staticmethod
    def test_single_prediction(K=4):

        k_nearest,df=KNearestTest.do_download()

        X_train, X_test, y_train, y_test, X_train_norm,X_test_norm=KNearestTest.split_datasets(k_nearest,df)

        model,y_pred,accuracy=KNearestTest.train_and_predict(k_nearest,X_train_norm,y_train,X_test_norm,y_test,k=K)

        print("Expected y_test:{}".format(y_test))
        print("Result y_pred:{}".format(y_pred))
        print("Accuracy:{}".format(accuracy))

    @staticmethod
    def show_range_predictions():

        Ks=range(1,11)#1 to 10
        mean_acc = np.zeros((len(Ks) - 1))
        std_acc = np.zeros((len(Ks) - 1))

        k_nearest_algo, df = KNearestTest.do_download()

        X_train, X_test, y_train, y_test, X_train_norm, X_test_norm = KNearestTest.split_datasets(k_nearest_algo, df)

        for n in range(1, len(Ks)):
            # Train Model and Predict
            model, y_pred, accuracy = KNearestTest.train_and_predict(k_nearest_algo, X_train_norm, y_train, X_test_norm,
                                                                     y_test, k=n)
            mean_acc[n - 1] = accuracy

            std_acc[n - 1] = np.std(y_pred == y_test) / np.sqrt(y_pred.shape[0])

        k_nearest_algo.plot_k_vs_acc(len(Ks),mean_acc,std_acc)
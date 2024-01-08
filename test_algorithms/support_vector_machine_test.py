import numpy as np

from classification.support_vector_machine import SupportVectorMachine
from util.file_downloader import FileDownloader


from util.plotter import Plotter


class SupportVectorMachineTest():
    output_path = "./input/support_vector_machine.csv"

    download_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv"

    def __init__(self):
        pass

    @staticmethod
    def initialize():
        svm = SupportVectorMachine()
        return svm

    @staticmethod
    def do_download():
        downloader = FileDownloader()
        downloader.download(SupportVectorMachineTest.download_url, SupportVectorMachineTest.output_path)
        svm = SupportVectorMachine()
        df = svm.load_data(SupportVectorMachineTest.output_path)
        return svm, df

    @staticmethod
    def split_datasets(svm,cell_df):

        X= svm.get_X_axis_as_arr(cell_df,['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit'])

        Y=svm.get_Y_axis_as_arr(cell_df,"Class","int")

        # # we split the dataset
        X_train, X_test, y_train, y_test = svm.split_data(X, Y)
        return  X_train, X_test, y_train, y_test



    @staticmethod
    def test():
        svm, cell_df = SupportVectorMachineTest.do_download()

        #svm.plot_classification(cell_df,"Clump","UnifSize","Class","malignant","benign",max_samples=50)

        #1- Do I need to convert SVM columns to numeric values?
        #cell_df=svm.clean_dataset(cell_df,"BareNuc","int")

        X_train, X_test, y_train, y_test = SupportVectorMachineTest.split_datasets(svm,cell_df)

        clf=svm.train_model(X_train,y_train)#clf is the model trained

        y_pred=svm.predict(clf,X_test)

        print("Prediction successfully finished...")
        print("Jaccard index:{}".format(svm.jaccard_index(y_test, y_pred,pos_label=2)))
        print("F1 Score:{}".format(svm.f1_score_index(y_test, y_pred)))

        conf_matrix = svm.get_confussion_matrix(y_test, y_pred, labels=[2,4])

        Plotter.plot_confussion_matrix(conf_matrix, classes=['Benign(2)','Malignant(4)'], normalize=False,
                                       title='Confusion matrix')





# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pip._vendor.distlib.compat import raw_input

from classification.decission_tree import DecissionTree
from classification.k_nearest_neighbour import KNearestNeighbour
from test_algorithms.decission_tree_test import DecissionTreeTest
from test_algorithms.k_nearest_test import KNearestTest
from test_algorithms.logistic_regression_test import LogisticRegressionTest
from test_algorithms.support_vector_machine_test import SupportVectorMachineTest
from util.file_downloader import FileDownloader


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def k_nearest_tests():
    #K-Neaerest Neighbour
    KNearestTest.test_single_prediction(K=4)

    # K-Nearest - Multiple Ks
    # KNearestTest.show_range_predictions()

def decission_tree_tests():
    # Decission Trees
    DecissionTreeTest.test_single_prediction()

def logistic_regression_test():
    #Logistic Regression
    LogisticRegressionTest.test()

def support_vector_machine_test():
    # Logistic Regression
    SupportVectorMachineTest.test()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #k_nearest_tests()
    #decission_tree_tests()
    #logistic_regression_test()
    support_vector_machine_test()

    raw_input("Press Enter to continue...")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from classification.decission_tree import DecissionTree
from classification.k_nearest_neighbour import KNearestNeighbour
from test_algorithms.decission_tree_test import DecissionTreeTest
from test_algorithms.k_nearest_test import KNearestTest
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





# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #k_nearest_tests()
    decission_tree_tests()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

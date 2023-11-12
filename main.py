# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from classification.k_nearest_neighbour import KNearestNeighbour
from test_algorithms.k_nearest_test import KNearestTest
from util.file_downloader import FileDownloader


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #K-Neaerest Neighbour
    #KNearestTest.test_single_prediction(K=4)


    #K-Nearest - Multiple Ks
    KNearestTest.show_range_predictions()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

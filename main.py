# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pip._vendor.distlib.compat import raw_input

from classification.decission_tree import DecissionTree
from classification.k_nearest_neighbour import KNearestNeighbour
from exams.module_7_final_exam import Module7Exam
from test_algorithms.decission_tree_test import DecissionTreeTest
from test_algorithms.k_means_test import KMeansTest
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


def k_means_test():
    KMeansTest.test()
    #KMeansTest.draw_random_data()

def module7_test():
    exam=Module7Exam()
    df =exam.q1_display_data_types()

    #exam.q2_drop_and_describe(df)
    #exam.q2_extra_analysis(df)
    #exam.q3_count_unique_floor_values(df)
    #exam.q4_boxplot_eval_outliers(df)
    #exam.q5_regplot_eval(df)
    #exam.q6_fit_linear_regression_model(df)
    #exam.q6_bis_fit_linear_regression_model_only_R2(df)
    #exam.q7_fit_multilinear_regr_model(df)
    #exam.q8_pipeline_evaluation(df)
    #exam.q9_ridge_eval_model(df)
    exam.q10_ridge_w_plynomial_transform(df)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #k_nearest_tests()
    #decission_tree_tests()
    #logistic_regression_test()
    #support_vector_machine_test()
    #k_means_test()
    module7_test()

    raw_input("Press Enter to continue...")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

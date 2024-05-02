# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pip._vendor.distlib.compat import raw_input

from coursera_course_ML.exams.data_science_IBM_cert.exams.exam10_presentation_calculations import \
    Exam10PresentationCalculations
from coursera_course_ML.exams.data_science_IBM_cert.exams.exam_5_final_exam import Exam5FinalExam
from coursera_course_ML.exams.data_science_IBM_cert.exams.exam_8_dashboard import Module8Dashboard
from coursera_course_ML.exams.data_science_IBM_cert.exams.module_7_final_exam import Module7Exam
from coursera_course_ML.exams.deep_learning_specialization.neural_networks_week2_test import NeuralNetworksWeek2Test
from coursera_course_ML.exams.hidden_road_test.hidden_road_test import HiddenRoadTest
from coursera_course_ML.excercises.course10_space_y_machine_learning_algorithms_comparisson import Course10SpaceYMachineLearningPrediction
from coursera_course_ML.test_algorithms.ARIMA_tests import ARIMATests
from coursera_course_ML.test_algorithms.decission_tree_test import DecissionTreeTest
from coursera_course_ML.test_algorithms.k_means_test import KMeansTest
from coursera_course_ML.test_algorithms.k_nearest_test import KNearestTest
from coursera_course_ML.test_algorithms.logistic_regression_test import LogisticRegressionTest
from coursera_course_ML.test_algorithms.support_vector_machine_test import SupportVectorMachineTest


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
    #exam.q10_ridge_w_plynomial_transform(df)


def module8_test():
    exam = Module8Dashboard()
    #exam.task_1_1_create_sales_x_year_lineplot()
    #exam.task_1_2_sales_per_vehicle_type_lineplot()
    #exam.task_1_3_sales_recession_no_recession_lineplot()
    #exam.task_1_3_v2_sales_per_vehicle_type()
    #exam.task_1_4_sales_per_vehicle_type()
    #exam.task_1_5_sales_seasonality()
    #exam.task_1_6_consumer_conf_to_sales()
    #exam.task_1_6_avg_px_sales_volume()
    #exam.task_1_7_recession_non_recession_advertising()
    #exam.task_1_8_adv_expenditures_during_recessioln()
    #exam.task_1_9_unempl_and_vehicle_type_sales()
    exam.init_dashboard()
    pass

def course_10_week_1_practice():
    #app = Course10SpaceYDataCollectionTest()
    #app.week_1_test()

    #app=Course10SpaceYDataWranglingTest()
    #app.data_wrangling_test()

    #app=Course10SpaceYDataVisualization()
    #app.download_and_display()

    #app=Course10SpaceYDashboard()
    #app.init_dashboard()

    app=Course10SpaceYMachineLearningPrediction()
    #app.compare_ML_algorithms()
    app.custom_calculations()

    #app = Exam10PresentationCalculations()
    #app.save_launches_to_csv()
    pass
def exam_10_presentation_calculations():
    app=Exam10PresentationCalculations()
    #app.save_launches_to_csv()
    #app.slide_25_flight_number_vs_launch_site()
    #app.slide_26_payload_vs_launch_site()
    #app.slide_27_orbit_vs_success()
    #app.slide_28_flight_number_vs_orbit_type()
    #app.slide_30_payload_vs_orbit_type()
    #app.slide_32_launch_success_by_year()
    #app.slide_34_all_launch_site_names()
    #app.slide_35_filter_launch_sites()
    #app.slide_36_total_payload_mass()
    #app.slide_37_avg_payload_mass_by_f9_v1_1()
    #app.slide_38_first_succesful_ground_landing()
    #app.slide_39_succesfull_landing_with_specific_payload()
    #app.slide_40_succesful_vs_failed_landings()
    #app.slide_41_boost_w_payload_mas()
    #app.slide_42_failed_launches_for_2015()
    app.slide_43_rank_landing_outcomes()


def exam_5_stocks_analysis():
    app=Exam5FinalExam()
    app.init_dashboard()
    #app=StocksAnalysisSummary()
    #app.q1_extract_stock_price_w_yahoo()
    #app.q3_extract_stock_price_w_yahoo()

def ARIMA_tests():
    ARIMATests.test_stock_predictions()


def NumPy_tests():
    #NumpyTest.vectorize_array_test()
    #NumpyTest.normalize_rows_test()
    #NumpyTest.softmax_test()
    NeuralNetworksWeek2Test.test()

def HiddenRoadTestStart():
    HiddenRoadTest.test()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #k_nearest_tests()
    #decission_tree_tests()
    logistic_regression_test()
    #support_vector_machine_test()
    #k_means_test()
    #module7_test()
    #module8_test()
    #course_10_week_1_practice()
    #exam_10_presentation_calculations()
    #exam_5_stocks_analysis()
    #ARIMA_tests()
    #NumPy_tests
    #HiddenRoadTestStart()
    raw_input("Press Enter to continue...")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

from datetime import timedelta

from data_access_layer.date_range_classification_manager import DateRangeClassificationManager
from data_access_layer.economic_series_manager import EconomicSeriesManager

from framework.common.logger.message_type import MessageType
from logic_layer.ARIMA_models_analyzer import ARIMAModelsAnalyzer
from logic_layer.data_set_builder import DataSetBuilder
from logic_layer.ml_models_analyzer import MLModelAnalyzer


class DataManagement:

    def __init__(self,hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger):

        self.logger=logger

        self.data_set_builder=DataSetBuilder(hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger)


    def evaluate_algorithms_performance(self,series_csv,d_from,d_to):

        try:
            series_df=self.data_set_builder.build_series(series_csv,d_from,d_to)
            mlAnalyzer=MLModelAnalyzer(self.logger)
            comp_df= mlAnalyzer.fit_and_evaluate(series_df, DataSetBuilder._CLASSIFICATION_COL)
            return comp_df

        except Exception as e:
            msg="CRITICAL ERROR processing model @evaluate_algorithms_performance:{}".format(str(e))
            self.logger.do_log(msg,MessageType.ERROR)
            raise Exception(msg)

    def evaluate_trading_performance(self,symbol,series_csv,d_from,d_to,bias):

        try:
            symbol_df = self.data_set_builder.build_series(symbol,d_from,d_to)
            series_df = self.data_set_builder.build_series(series_csv, d_from, d_to)
            mlAnalyzer = MLModelAnalyzer(self.logger)
            portf_pos_dict = mlAnalyzer.evaluate_trading_performance_last_model(symbol_df,symbol,series_df, bias)
            return portf_pos_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @evaluate_trading_performance:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def run_predictions_last_model(self,series_csv,d_from,d_to):

        try:
            series_df = self.data_set_builder.build_series(series_csv, d_from, d_to,add_classif_col=False)
            mlAnalyzer = MLModelAnalyzer(self.logger)
            pred_dict = mlAnalyzer.run_predictions_last_model(series_df)
            return pred_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @run_predictions_last_model:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def build_ARIMA(self,symbol, period, d_from, d_to):
        try:
            series_df = self.data_set_builder.build_series(symbol, d_from, d_to, add_classif_col=False)
            arima_Analyzer = ARIMAModelsAnalyzer(self.logger)
            dickey_fuller_test_dict=arima_Analyzer.build_ARIMA_model(series_df,symbol,period,True)
            return dickey_fuller_test_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @build_ARIMA:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)


    def predict_ARIMA(self,symbol, p,d,q,d_from,d_to,period, steps):
        try:
            series_df = self.data_set_builder.build_series(symbol, d_from, d_to, add_classif_col=False)
            arima_Analyzer = ARIMAModelsAnalyzer(self.logger)
            preds=arima_Analyzer.build_and__predict_ARIMA_model(series_df,symbol,p,d,q,period,steps)
            return preds

        except Exception as e:
            msg = "CRITICAL ERROR processing model @predict_ARIMA:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)



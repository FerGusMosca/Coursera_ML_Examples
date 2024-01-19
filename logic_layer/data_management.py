from datetime import timedelta

from data_access_layer.date_range_classification_manager import DateRangeClassificationManager
from data_access_layer.economic_series_manager import EconomicSeriesManager
import pandas as pd

from framework.common.logger.message_type import MessageType
from logic_layer.ml_models_analyzer import MLModelAnalyzer


class DataManagement:
    _1_DAY_INTERVAL="1 day"
    _CLASSIFICATION_COL="classification"

    def __init__(self,hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger):
        self.hist_data_conn_str=hist_data_conn_str
        self.ml_reports_conn_str=ml_reports_conn_str
        self.classification_map_key=p_classification_map_key
        self.logger=logger
        self.economic_series_mgr= EconomicSeriesManager(hist_data_conn_str)
        self.date_range_classification_mgr=DateRangeClassificationManager(ml_reports_conn_str)
        self.classification_map_values=[]

    def get_extreme_dates(self,series_data_dict):
        min_date=None
        max_date=None


        for economic_value_list in series_data_dict.values():
            for econ_value in  economic_value_list:
                if min_date is None or min_date>econ_value.date:
                    min_date=econ_value.date

                if max_date is None or max_date<econ_value.date:
                    max_date=econ_value.date

        return  min_date,max_date


    def build_empty_dataframe(self,series_data_dict):

        column_list=["date"]

        for key in series_data_dict.keys():
            column_list.append(key)

        df = pd.DataFrame(columns=column_list)

        return df

    def eval_all_values_none(self,curr_date_dict):
        not_none=False

        for value in curr_date_dict.values():
            if value is not None:
                not_none=True

        return not not_none

    def assign_classification(self,curr_date):
        classification_value = next((classif_value for classif_value in self.classification_map_values
                                     if classif_value.date_start.date() <= curr_date.date() <= classif_value.date_end.date()), None)

        if classification_value is None:
            raise Exception("Could not find the proper classification for date {}".format(curr_date.date()))

        return  classification_value.classification



    def fill_dataframe(self,series_df,min_date,max_date,series_data_dict,add_classif_col=True):
        curr_date=min_date
        self.logger.do_log("Building the input dataframe from {} to {}".format(min_date.date(),max_date.date()), MessageType.INFO)
        while curr_date<=max_date:
            curr_date_dict={}
            for key in series_data_dict.keys():
                series_value=next((x for x in series_data_dict[key] if x.date == curr_date), None)
                if series_value is not None and series_value.close is not None:
                    curr_date_dict[key]=series_value.close
                else:
                    curr_date_dict[key]=None

            if not self.eval_all_values_none(curr_date_dict):
                curr_date_dict["date"]=curr_date

                if add_classif_col:
                    curr_date_dict[DataManagement._CLASSIFICATION_COL]=self.assign_classification(curr_date)

                self.logger.do_log("Adding dataframe row for date {}".format(curr_date.date()),MessageType.INFO)
                series_df=series_df.append(curr_date_dict, ignore_index=True)

            curr_date = curr_date + timedelta(days=1)

        self.logger.do_log("Input dataframe from {} to {} successfully created: {} rows".format(min_date, max_date,len(series_df)), MessageType.INFO)
        return series_df

    def load_classification_map_date_ranges(self):
        self.classification_map_values=self.date_range_classification_mgr.get_date_range_classification_values(self.classification_map_key)
        pass

    def     build_series(self,series_csv,d_from,d_to,add_classif_col=True):
        series_list = series_csv.split(",")

        series_data_dict = {}

        for serieID in series_list:
            economic_values = self.economic_series_mgr.get_economic_values(serieID, DataManagement._1_DAY_INTERVAL,
                                                                           d_from, d_to)
            series_data_dict[serieID] = economic_values

        min_date, max_date = self.get_extreme_dates(series_data_dict)

        series_df = self.build_empty_dataframe(series_data_dict)

        if add_classif_col:

            self.load_classification_map_date_ranges()

        series_df = self.fill_dataframe(series_df, min_date, max_date, series_data_dict,
                                        add_classif_col=add_classif_col)

        return  series_df

    def evaluate_algorithms_performance(self,series_csv,d_from,d_to):

        try:
            series_df=self.build_series(series_csv,d_from,d_to)
            mlAnalyzer=MLModelAnalyzer(self.logger)
            comp_df= mlAnalyzer.fit_and_evaluate(series_df, DataManagement._CLASSIFICATION_COL)
            return comp_df

        except Exception as e:
            msg="CRITICAL ERROR processing model @evaluate_algorithms_performance:".format(str(e))
            self.logger.do_log(msg,MessageType.ERROR)
            raise Exception(msg)

    def evaluate_trading_performance(self,symbol,series_csv,d_from,d_to,bias):

        try:
            symbol_df = self.build_series(symbol,d_from,d_to)
            series_df = self.build_series(series_csv, d_from, d_to)
            mlAnalyzer = MLModelAnalyzer(self.logger)
            portf_pos_dict = mlAnalyzer.evaluate_trading_performance_last_model(symbol_df,symbol,series_df, bias)
            return portf_pos_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @evaluate_trading_performance:".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def run_predictions_last_model(self,series_csv,d_from,d_to):

        try:
            series_df = self.build_series(series_csv, d_from, d_to,add_classif_col=False)
            mlAnalyzer = MLModelAnalyzer(self.logger)
            pred_dict = mlAnalyzer.run_predictions_last_model(series_df)
            return pred_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @run_predictions_last_model:".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)



from data_access_layer.economic_series_manager import EconomicSeriesManager


class DataManagement:
    _1_DAY_INTERVAL="1 day"
    def __init__(self,conn_str,logger):
        self.conn_str=conn_str
        self.logger=logger
        self.economic_series_mgr= EconomicSeriesManager(conn_str)



    def build_dataframes(self,series_csv,d_from,d_to):

        #TODO log///TryCatch
        series_list=series_csv.split(",")

        series_data_dict={}

        for serieID in series_list:
            economic_values=self.economic_series_mgr.get_economic_values(serieID,DataManagement._1_DAY_INTERVAL,d_from,d_to)
            series_data_dict[serieID]=economic_values

        #TODO--> Take everything to a single dataframe!
        #TODO --> Later --> the classification stuff
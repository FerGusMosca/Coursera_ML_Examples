from common.util.date_handler import DateHandler
from common.util.logger import Logger
from common.util.ml_settings_loader import MLSettingsLoader
from framework.common.logger.message_type import MessageType
from logic_layer.data_management import DataManagement
from IPython.display import display
import pandas as pd
_DATE_FORMAT="%m/%d/%Y"


last_trading_dict=None

def show_commands():

    print("#1-TrainAlgos [SeriesCSV] [from] [to]")
    print("#3-RunPredictionsLastModel [SeriesCSV] [from] [to]")
    print("#4-EvalBiasedTradingAlgo [Symbol] [SeriesCSV] [from] [to] [Bias]")
    print("#5-EvaluateARIMA [Symbol] [Period] [from] [to]")
    print("#6-PredictARIMA [Symbol] [p] [d] [q] [from] [to] [Period] [Step]")
    print("#7-EvalSingleIndicatorAlgo` [Symbol] [indicator] [from] [to] [inverted]")
    print("#8-EvalMLBiasedAlgo [Symbol] [indicator] [SeriesCSV] [from] [to] [inverted]")
    print("#9-TrainDeepNeuralNetwork [true_path] [false_path] [true_lavel] [learning_rate] [iterations] [arch_file] [activ_file] [output file]")

    print("#n-Exit")

def params_validation(cmd,param_list,exp_len):
    if(len(param_list)!=exp_len):
        raise Exception("Command {} expects {} parameters".format(cmd,exp_len))


def process_train_algos(cmd_param_list,str_from,str_to):
    loader=MLSettingsLoader()
    logger=Logger()
    try:
        logger.print("Initializing dataframe creation for series : {}".format(cmd_param_list[1]),MessageType.INFO)

        config_settings=loader.load_settings("./configs/commands_mgr.ini")

        dataMgm= DataManagement(config_settings["hist_data_conn_str"],config_settings["ml_reports_conn_str"],
                                config_settings["classification_map_key"], logger)
        dataMgm.train_algos(cmd_param_list, DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                            DateHandler.convert_str_date(str_to, _DATE_FORMAT))

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)),MessageType.ERROR)

def process_biased_trading_algo(symbol, cmd_series_csv,str_from,str_to,bias):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        global last_trading_dict
        logger.print("Evaluating trading performance for symbol from last model from {} to {}".format(str_from, str_to), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = DataManagement(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                 config_settings["classification_map_key"], logger)
        summary_dict = dataMgm.evaluate_trading_performance(symbol,cmd_series_csv,
                                                       DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                                       DateHandler.convert_str_date(str_to, _DATE_FORMAT),bias,
                                                       last_trading_dict)

        last_trading_dict=summary_dict

        print("Displaying all the different models predictions for the different alogs:")

        for key in summary_dict.keys():
            print("============{}============ for {}".format(key,symbol))
            summary=summary_dict[key]
            print("From={} To={}".format(str_from, str_to))
            print("Nom. Profit={}".format(summary.calculate_th_nom_profit()))
            print("Pos. Size={}".format(summary.portf_pos_size))


    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)




def process_run_preductions_last_model(cmd_param_list,str_from,str_to):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Running predictions fro last model from {} to {}".format(str_from,str_to), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = DataManagement(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                 config_settings["classification_map_key"], logger)
        pred_dict=dataMgm.run_predictions_last_model(cmd_param_list,
                                                           DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                                           DateHandler.convert_str_date(str_to, _DATE_FORMAT))
        print("Displaying all the different models predictions for the different alogs:")
        pd.set_option('display.max_rows', None)
        for key in pred_dict.keys():
            print("============{}============".format(key))
            display(pred_dict[key])
            print("")
            print("")
        pd.reset_option('display.max_rows')

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_eval_ARIMA(symbol,period, str_from,str_to):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Building ARIMA model for {} (period {}) from {} to {}".format(symbol,period, str_from, str_to), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = DataManagement(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                 config_settings["classification_map_key"], logger)
        pred_dict = dataMgm.build_ARIMA(symbol,period,
                                       DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                       DateHandler.convert_str_date(str_to, _DATE_FORMAT))

        print("======= Showing Dickey Fuller Test after building ARIMA======= ")
        for key in pred_dict.keys():
            print("{}={}".format(key,pred_dict[key]))


        pass #brkpnt to see the graph!

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)

        # [Symbol] [indicator] [from] [to] [inverted]
def process_eval_single_indicator_algo(symbol,indicator,str_from,str_to,inverted):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Evaluating Single Indicator Algo for {} from {} to {}".format(symbol, str_from, str_to), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = DataManagement(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                 config_settings["classification_map_key"], logger)
        portf_summary = dataMgm.eval_singe_indicator_algo(symbol,indicator,inverted=="True",
                                       DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                       DateHandler.convert_str_date(str_to, _DATE_FORMAT))


        print("============= Displaying porftolio status for symbol {} =============".format(symbol))
        print("From={} To={}".format(str_from,str_to))
        print("Nom. Profit={}".format(portf_summary.calculate_th_nom_profit()))
        print("Pos. Size={}".format(portf_summary.portf_pos_size))
        pass

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)

def process_eval_ml_biased_algo(symbol, indicator,seriesCSV,str_from,str_to,inverted):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Evaluating ML biasde algo for {} from {} to {}".format(symbol, str_from, str_to),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = DataManagement(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                 config_settings["classification_map_key"], logger)
        portf_summary_dict = dataMgm.eval_ml_biased_algo(symbol,indicator,seriesCSV,
                                                          DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                                          DateHandler.convert_str_date(str_to, _DATE_FORMAT),
                                                          inverted == "True")

        for algo in portf_summary_dict.keys():
            portf_summary=portf_summary_dict[algo]
            print("============= Displaying porftolio status for symbol {} w/Algo {} =============".format(symbol,algo))
            print("From={} To={}".format(str_from, str_to))
            print("Nom. Profit={}".format(portf_summary.calculate_th_nom_profit()))
            print("Pos. Size={}".format(portf_summary.portf_pos_size))
            print( "============= =============")

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_train_deep_neural_network(true_path,false_path,true_label,learning_rate,iterations,arch_file, activ_file,output_file):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Train Deep Neural Network",MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = DataManagement(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                 config_settings["classification_map_key"], logger)
        dataMgm.train_deep_neural_network(true_path, false_path, true_label,learning_rate,iterations,arch_file, activ_file,output_file)

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_predict_ARIMA(symbol, p,d,q, str_from,str_to,period,step):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Predicting w/last built ARIMA model for {} (period {}) from {} to {}".format(symbol,period, str_from, str_to), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = DataManagement(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                 config_settings["classification_map_key"], logger)
        preds_list = dataMgm.predict_ARIMA(symbol,int(p),int(d),int(q),
                                       DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                       DateHandler.convert_str_date(str_to, _DATE_FORMAT),
                                      period,int(step))
        print("==== Displaying Predictions for following periods ==== ")
        i=1
        for pred in preds_list:
            print("{} --> {} %".format(period+str(i),"{:.2f}".format(pred*100)))
            i+=1

        pass #brkpnt to see the graph!

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_commands(cmd):

    cmd_param_list=cmd.split(" ")

    if cmd_param_list[0]=="TrainAlgos":
        params_validation("TrainAlgos", cmd_param_list, 4)
        process_train_algos(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3])

    elif cmd_param_list[0]=="RunPredictionsLastModel":
        params_validation("RunPredictionsLastModel", cmd_param_list, 4)
        process_run_preductions_last_model( cmd_param_list[1], cmd_param_list[2], cmd_param_list[3])
    elif cmd_param_list[0]=="EvalBiasedTradingAlgo":
        params_validation("EvalBiasedTradingAlgo", cmd_param_list, 6)
        process_biased_trading_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                    cmd_param_list[5])
    elif cmd_param_list[0] == "EvaluateARIMA":
        params_validation("EvaluateARIMA", cmd_param_list, 5)
        process_eval_ARIMA(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4])
    elif cmd_param_list[0] == "PredictARIMA":
        params_validation("PredictARIMA", cmd_param_list, 9)
        process_predict_ARIMA(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                              cmd_param_list[5], cmd_param_list[6], cmd_param_list[7], cmd_param_list[8])
    elif cmd_param_list[0] == "EvalSingleIndicatorAlgo":
        params_validation("EvalSingleIndicatorAlgo", cmd_param_list, 6)
        process_eval_single_indicator_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                            cmd_param_list[5])
    elif cmd_param_list[0] == "EvalMLBiasedAlgo":
        params_validation("EvalMLBiasedAlgo", cmd_param_list, 7)
        process_eval_ml_biased_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                            cmd_param_list[5],cmd_param_list[6])

    elif cmd_param_list[0] == "TrainDeepNeuralNetwork":
        params_validation("TrainDeepNeuralNetwork", cmd_param_list, 9)
        process_train_deep_neural_network(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3],float( cmd_param_list[4]),
                                          int( cmd_param_list[5]),cmd_param_list[6],cmd_param_list[7],cmd_param_list[8])


    else:
        print("Not recognized command {}".format(cmd_param_list[0]))


if __name__ == '__main__':

    while True:

        show_commands()
        cmd=input("Enter a command:")
        try:
            process_commands(cmd)
            if(cmd=="Exit"):
                break
        except Exception as e:
            print("Could not process command:{}".format(str(e)))


    print("Exit")

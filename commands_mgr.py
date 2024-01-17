from common.util.date_handler import DateHandler
from common.util.logger import Logger
from common.util.settings_loader import SettingsLoader
from framework.common.logger.message_type import MessageType
from logic_layer.data_management import DataManagement

_DATE_FORMAT="%m/%d/%Y"

def show_commands():

    print("#1-EvaluateAlgos [SeriesCSV]")
    print("#n-Exit")

def params_validation(cmd,param_list,exp_len):
    if(len(param_list)!=exp_len):
        raise Exception("Command {} expects {} parameters".format(cmd,exp_len))


def process_evaluate_algos(cmd_param_list,str_from,str_to):
    loader=SettingsLoader()
    logger=Logger()
    try:
        logger.print("Initializing dataframe creation for series : {}".format(cmd_param_list[1]),MessageType.INFO)

        config_settings=loader.load_settings("./configs/commands_mgr.ini")

        dataMgm= DataManagement(config_settings["hist_data_conn_str"],config_settings["ml_reports_conn_str"],
                                config_settings["classification_map_key"], logger)
        dataMgm.evaluate_algorithms_performance(cmd_param_list, DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                                DateHandler.convert_str_date(str_to, _DATE_FORMAT))
        #TODO print ouput result
    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)),MessageType.ERROR)


def process_evaluate_algos_last_model(cmd_param_list,str_from,str_to):
    loader=SettingsLoader()
    logger=Logger()
    try:
        logger.print("Initializing dataframe creation for series : {}".format(cmd_param_list[1]),MessageType.INFO)

        config_settings=loader.load_settings("./configs/commands_mgr.ini")

        dataMgm= DataManagement(config_settings["hist_data_conn_str"],config_settings["ml_reports_conn_str"],
                                config_settings["classification_map_key"], logger)
        dataMgm.evaluate_algorithms_performance_last_model(cmd_param_list, DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                                DateHandler.convert_str_date(str_to, _DATE_FORMAT))
        #TODO print ouput result
    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)),MessageType.ERROR)




def process_commands(cmd):

    cmd_param_list=cmd.split(" ")

    if cmd_param_list[0]=="EvaluateAlgos":
        params_validation("EvaluateAlgos", cmd_param_list, 4)
        process_evaluate_algos(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3])
        #EvaluateAlgosLastModel
    elif cmd_param_list[0]=="EvaluateAlgosLastModel":
        params_validation("EvaluateAlgosLastModel", cmd_param_list, 4)
        process_evaluate_algos_last_model(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3])
        #
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

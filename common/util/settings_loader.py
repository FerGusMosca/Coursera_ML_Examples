import configparser

class SettingsLoader():

    def __init__(self):
        pass


    def load_settings(self,config_file_path):
        try:
            config = configparser.ConfigParser()
            config.read(config_file_path)
            config_settings={}


            config_settings["connection_string"]= config['DB']['CONNECTION_STRING']


            return  config_settings



        except Exception as e:
            raise Exception("Critical error reading config file {}:{}".format(config_file_path,str(e)))
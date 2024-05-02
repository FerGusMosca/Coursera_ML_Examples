import datetime

import requests
import json
import pandas as pd
from pandas.io.json import json_normalize

from service_layer.launches_service_client import LaunchesServiceClient


class Course10SpaceYDataCollectionTest:

    #region Constructor


    def __init__(self):
        # Setting this option will print all collumns of a dataframe
        pd.set_option('display.max_columns', None)
        # Setting this option will print all of the data in a feature
        pd.set_option('display.max_colwidth', None)

    #endregion

    #region Public Methods

    def week_1_test(self):

        svc_client=LaunchesServiceClient()
        launch_pd=svc_client.get_all_launches()

        launch_falcon_9 = launch_pd[launch_pd["BoosterVersion"]=="Falcon 9"]

        #2-Number of Falcon's 9
        print("We have {} Falcon 9 rows".format(len(launch_falcon_9)))

        missing_landing_pad_pd=launch_falcon_9[launch_falcon_9["LandingPad"].isnull()]
        #3-Missing Landing Pads
        print("Missing LandingPads (LandingPad=None): {}".format(len(missing_landing_pad_pd)))


    #endregion






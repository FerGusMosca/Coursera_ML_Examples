import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from service_layer.launches_service_client import LaunchesServiceClient


class Exam10PresentationCalculations:
    _INPUT_FOLDER="./input/"
    _CSV_ALL_LAUNCHES_PATH = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
    _CSV_FALCON_9_PATH = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv'
    _CSV_PATH_Q = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv"


    def __init__(self):
        # Setting this option will print all collumns of a dataframe
        pd.set_option('display.max_columns', None)
        # Setting this option will print all of the data in a feature
        pd.set_option('display.max_colwidth', None)

    def save_launches_to_csv(self):
        svc_client = LaunchesServiceClient()
        df = svc_client.get_all_launches()
        df.to_csv("{}space_x_launches.csv".format(Exam10PresentationCalculations._INPUT_FOLDER))

    def get_launches_from_csv(self):
        df = pd.read_csv("{}space_x_launches.csv".format(Exam10PresentationCalculations._INPUT_FOLDER))
        return df


    def slide_25_flight_number_vs_launch_site(self):

        svc_client = LaunchesServiceClient()
        df = svc_client.get_all_launches()

        size = df['LaunchSite']
        sns.scatterplot(data=df, x='FlightNumber', y='LaunchSite', size=size)
        plt.xlabel("Flight Number", fontsize=18)
        plt.ylabel("Launch Site", fontsize=16)
        plt.title('Flight Number vs Launch Site')
        plt.show()

    def slide_26_payload_vs_launch_site(self):
        df = self.get_launches_from_csv()
        size = df['LaunchSite']
        sns.scatterplot(data=df, x='LaunchSite', y='PayloadMass',size=size)
        plt.xlabel("Launch Site", fontsize=18)
        plt.ylabel("Payload Mass (kg)", fontsize=16)
        plt.title('Payload Mass vs Launch Site')
        plt.show()

    def slide_27_orbit_vs_success(self):
        df_launches = self.get_launches_from_csv()

        succesful_landings_df=df_launches[(df_launches["Outcome"]=="True Ocean")
                                           |(df_launches["Outcome"]=="True RTLS")
                                           |(df_launches["Outcome"]=="True ASDS")].groupby("Orbit").count().reset_index()

        succesful_summary_df=succesful_landings_df[["Orbit","FlightNumber"]].rename(columns={'FlightNumber': 'Success'})

        unuccesful_landings_df = df_launches[(df_launches["Outcome"] != "True Ocean")
                                            & (df_launches["Outcome"] != "True RTLS")
                                            & (df_launches["Outcome"] != "True ASDS")].groupby("Orbit").count().reset_index()
        unsuccesful_summary_df = unuccesful_landings_df[["Orbit", "FlightNumber"]].rename(columns={'FlightNumber': 'Failure'})

        summary_df=succesful_summary_df.merge(unsuccesful_summary_df,on="Orbit")


        summary_df=summary_df.melt(id_vars='Orbit',value_name='Res', var_name='Result')

        sns.barplot(x=summary_df["Orbit"], y="Res",hue="Result", data=summary_df)
        plt.xlabel('Orbit')
        plt.ylabel('Sucess/Failures')
        plt.title('Success/Failure per Orbit')
        plt.show()
        pass

    def slide_28_flight_number_vs_orbit_type(self):
        df_launches = self.get_launches_from_csv()

        size = df_launches['LaunchSite']
        sns.scatterplot(data=df_launches, x='FlightNumber', y='Orbit', size=size)
        plt.xlabel("Flight Number", fontsize=18)
        plt.ylabel("Orbit", fontsize=16)
        plt.title('Flight Number vs Orbit')
        plt.show()

    def slide_30_payload_vs_orbit_type(self):
        df_launches = self.get_launches_from_csv()

        size = df_launches['LaunchSite']
        sns.scatterplot(data=df_launches, x='Orbit', y='PayloadMass', size=size)
        plt.xlabel("Orbit", fontsize=18)
        plt.ylabel("Payload Mass (Kg)", fontsize=16)
        plt.title('Orbit vs Payload Mass')
        plt.show()

    def slide_32_launch_success_by_year(self):
        df_launches = self.get_launches_from_csv()

        succesful_landings_by_year_df = df_launches[(df_launches["Outcome"] == "True Ocean")
                                            | (df_launches["Outcome"] == "True RTLS")
                                            | (df_launches["Outcome"] == "True ASDS")]\

        succesful_landings_by_year_df['Date'] = pd.to_datetime(succesful_landings_by_year_df['Date'])

        success_land_by_year_df=succesful_landings_by_year_df.groupby(succesful_landings_by_year_df['Date'].dt.year).count()
        #success_land_by_year_df=success_land_by_year_df.rename(columns={'': 'Year'})
        #new_col_map = {0: 'Year'}
        #success_land_by_year_df.columns = [new_col_map.get(i, col) for i, col in enumerate(success_land_by_year_df.columns)]

        #TODO -->GET THE FUCKING YEARS!

        size = df_launches['LaunchSite']
        sns.lineplot(data=success_land_by_year_df, x='Year', y='FlightNumber', size=size)
        plt.xlabel("Orbit", fontsize=18)
        plt.ylabel("Payload Mass (Kg)", fontsize=16)
        plt.title('Orbit vs Payload Mass')
        plt.show()

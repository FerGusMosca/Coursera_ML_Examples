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

        succesful_landings_by_year_df['Year'] = pd.to_datetime(succesful_landings_by_year_df['Date']).dt.year

        success_land_by_year_df=succesful_landings_by_year_df.groupby("Year")\
                                         .count().reset_index().rename(columns={'FlightNumber': 'SuccesfulLaunches'})

        success_land_by_year_df=success_land_by_year_df[["Year","SuccesfulLaunches"]]


        sns.lineplot(data=success_land_by_year_df, x='Year', y='SuccesfulLaunches')
        plt.xlabel("Year", fontsize=18)
        plt.ylabel("Successful Launches", fontsize=16)
        plt.title('Launch Success Yearly Trend')
        plt.show()

    def slide_34_all_launch_site_names(self):
        df_launches = self.get_launches_from_csv()
        df_launches=df_launches.groupby("LaunchSite").count().reset_index()
        return  df_launches


    def slide_35_filter_launch_sites(self):
        df_launches = self.get_launches_from_csv()
        df_launches = df_launches[df_launches["LaunchSite"].str.startswith("CCS")]
        return  df_launches


    def slide_36_total_payload_mass(self):
        df_launches = self.get_launches_from_csv()
        df_launches=df_launches[df_launches["PayloadMass"].isnull()==False]
        total_payload_mass=df_launches["PayloadMass"].sum()
        print("Total Payload mass of all the launches is :{} kg".format(total_payload_mass))
        return total_payload_mass


    def slide_37_avg_payload_mass_by_f9_v1_1(self):
        df_launches = self.get_launches_from_csv()
        df_launches=df_launches[(df_launches["BoosterVersion"] == "Falcon 9") & (df_launches["PayloadMass"].isnull()==False)]
        df_avg_mass=df_launches["PayloadMass"].mean()
        print("The average payload mass for Falcon 9 booster is {} kg".format(round(df_avg_mass,2)))
        return df_avg_mass

    def slide_38_first_succesful_ground_landing(self):
        df_launches = self.get_launches_from_csv()
        #1- We filter all the succesful launches
        df_launches = df_launches[(df_launches["Outcome"] == "True Ocean")
                                                    | (df_launches["Outcome"] == "True RTLS")
                                                    | (df_launches["Outcome"] == "True ASDS")]

        #2- We properly format the column Date to be a datetime
        df_launches['Date'] = pd.to_datetime(df_launches['Date'])

        #3-We sort by date and we pick the first row
        first_launch = df_launches.sort_values(by='Date').iloc[0]["Date"]
        print("The first succesful launch of a rocker was made on {}".format(first_launch.strftime('%m-%d-%Y')))
        return  first_launch

    def slide_39_succesfull_landing_with_specific_payload(self):
        df_launches = self.get_launches_from_csv()
        #1- We filter all the succesful launches
        df_launches = df_launches[(df_launches["Outcome"] == "True Ocean")
                                                    | (df_launches["Outcome"] == "True RTLS")
                                                    | (df_launches["Outcome"] == "True ASDS")]

        # 2- Then we filter the payload mass
        df_launches=df_launches[(df_launches["PayloadMass"]>=4000)& (df_launches["PayloadMass"]<=6000)]
        print(df_launches)
        return  df_launches

    def slide_40_succesful_vs_failed_landings(self):
        df_launches = self.get_launches_from_csv()

        succesful_landings_df = df_launches[(df_launches["Outcome"] == "True Ocean")
                                            | (df_launches["Outcome"] == "True RTLS")
                                            | (df_launches["Outcome"] == "True ASDS")]



        unuccesful_landings_df = df_launches[(df_launches["Outcome"] != "True Ocean")
                                             & (df_launches["Outcome"] != "True RTLS")
                                             & (df_launches["Outcome"] != "True ASDS")]


        print("The total number of succesful landings is:{}".format(len(succesful_landings_df)))
        print("The total number of failed landings is:{}".format(len(unuccesful_landings_df)))

        return  succesful_landings_df,unuccesful_landings_df


    def slide_41_boost_w_payload_mas(self):
        df_launches = self.get_launches_from_csv()
        df_launches['Date'] = pd.to_datetime(df_launches['Date'])

        max_payload_df = df_launches.sort_values(by='PayloadMass', ascending=False).iloc[0][["Date","BoosterVersion","PayloadMass"]]
        print("THe heaviest payload was launched on {} with the Booster {} with a payload mass of {} kg"
              .format(max_payload_df.loc["Date"].strftime('%m-%d-%Y'),
                      max_payload_df.loc["BoosterVersion"],
                      max_payload_df.loc["PayloadMass"]))
        return  max_payload_df


    def slide_42_failed_launches_for_2015(self):
        df_launches = self.get_launches_from_csv()

        #1- First we filter the unsuccesful landings
        unsuccessful_drone_ships_df = df_launches[df_launches["Outcome"] == "False ASDS"]

        #2- THen we create a column w/ the year
        unsuccessful_drone_ships_df['Year'] = pd.to_datetime(unsuccessful_drone_ships_df['Date']).dt.year

        #3-We filter the year and print Booster Version and Launch Site
        print("Printing Booster Version and Launch Site of drone ship failed landings of 2015")
        print( unsuccessful_drone_ships_df[unsuccessful_drone_ships_df["Year"] == 2015][["BoosterVersion","LaunchSite"]])
        return unsuccessful_drone_ships_df

    def slide_43_rank_landing_outcomes(self):
        df_launches = self.get_launches_from_csv()
        df_launches['Date'] = pd.to_datetime(df_launches['Date'])

        filtered_launches_df=df_launches[(df_launches["Date"]>=pd.to_datetime('2010-06-04'))
                                         & (df_launches["Date"]<=pd.to_datetime('2017-03-20'))]

        filtered_launches_df = filtered_launches_df.groupby("Outcome")\
                                                   .count().reset_index()\
                                                   .rename(columns={'FlightNumber': 'Count'})[["Outcome","Count"]]\
                                                   .sort_values(by='Count', ascending=False)


        print("Landing Ranking between 2010-06-04 and 2017-03-20")
        print(filtered_launches_df)
        return  filtered_launches_df

import pandas as pd

class Course10SpaceYDataWranglingTest:


    _CSV_PATH="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv"

    def __init__(self):
        # Setting this option will print all collumns of a dataframe
        pd.set_option('display.max_columns', None)
        # Setting this option will print all of the data in a feature
        pd.set_option('display.max_colwidth', None)

    def data_wrangling_test(self):
        df = pd.read_csv(Course10SpaceYDataWranglingTest._CSV_PATH)

        #1-How many launches came from CCAFS SLC 40?
        launch_sites_df=df.groupby(["LaunchSite"],as_index=False).count()
        print("Launchs from CCAFS SLC 40{}".format(launch_sites_df[launch_sites_df["LaunchSite"]=="CCAFS SLC 40"]["Flights"][0]))

        #2-Successful lands
        succesfull_landings_df=df[(df["Outcome"]=="True Ocean") |(df["Outcome"]=="True RTLS")|(df["Outcome"]=="True ASDS")]
        print("Successful Landings:{}".format(len(succesfull_landings_df)))


        #3- How many landing outcomes in the column landing_outcomes had  a value of none
        orbit_counts_df = df.groupby(["Orbit"], as_index=False).count()
        print("GTO Orbits!:{}".format(orbit_counts_df[orbit_counts_df["Orbit"] == "GTO"]["Flights"].values[0]))

        #4-How many landing outcomes in the column landing_outcomes had  a value of none
        none_outcomes=len(df[df["Outcome"].isnull()==True])
        print("None Outcomes:{}".format(none_outcomes))

        pass
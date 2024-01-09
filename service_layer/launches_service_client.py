from pandas.io.json import json_normalize
import requests
import datetime
import pandas as pd

class LaunchesServiceClient:
    # region Consts
    _BASE_URL = "https://api.spacexdata.com/v4/{}"

    _ALL_LAUNCHES = "launches/past"

    _GET_ROCKET = "rockets/{}"

    _GET_LAUNCHPAD = "launchpads/{}"

    _GET_PAYLOAD = "payloads/{}"

    _GET_CORES = "cores/{}"

    _GET_ALL_LAUNCHES = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json"

    # endregion

    # region Service Layer

    def normalize_response(self, r):

        if r.status_code == 200:
            resp = r.text
            json = json_normalize(r.json())
            return json
        else:
            raise Exception("Error processing response:{}".format(r.text))

    def get_rocket(self, id):
        url = LaunchesServiceClient._BASE_URL.format(
            LaunchesServiceClient._GET_ROCKET.format((id)))
        r = requests.get(url)
        return self.normalize_response(r)

    def get_launchpad(self, id):
        url = LaunchesServiceClient._BASE_URL.format(
            LaunchesServiceClient._GET_LAUNCHPAD.format((id)))
        r = requests.get(url)
        return self.normalize_response(r)

    def get_payload(self, id):
        url = LaunchesServiceClient._BASE_URL.format(
            LaunchesServiceClient._GET_PAYLOAD.format((id)))
        r = requests.get(url)
        return self.normalize_response(r)

    def get_launches(self):

        url = LaunchesServiceClient._GET_ALL_LAUNCHES  # same as the prev
        r = requests.get(url)
        return self.normalize_response(r)

    def get_core(self, id):
        url = LaunchesServiceClient._BASE_URL.format(
            LaunchesServiceClient._GET_CORES.format((id)))
        r = requests.get(url)
        return self.normalize_response(r)

    # endregion

    #region Private Methods

    def format_launches(self, launches_df):
        launches_df = launches_df[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

        # We will remove rows with multiple cores because those are falcon rockets with 2 extra rocket
        # boosters and rows that have multiple payloads in a single rocket.
        launches_df = launches_df[launches_df['cores'].map(len) == 1]
        launches_df = launches_df[launches_df['payloads'].map(len) == 1]

        # Since payloads and cores are lists of size 1 we will also extract the single value in the list
        # and replace the feature.
        # summary: we have a list--> we will have the first element
        # in one case it is an id. On the other (cores) it is the first json element in the list
        launches_df['cores'] = launches_df['cores'].map(lambda x: x[0])
        launches_df['payloads'] = launches_df['payloads'].map(lambda x: x[0])

        # We also want to convert the date_utc to a datetime datatype and then extracting the date leaving the time
        launches_df['date'] = pd.to_datetime(launches_df['date_utc']).dt.date

        # Using the date we will restrict the dates of the launches
        launches_df = launches_df[launches_df['date'] <= datetime.date(2020, 11, 13)]
        return launches_df

    def get_rocket_name_col(self, launches_df):
        rocket_name = []
        rocket_dict = {}
        for rocket_id in launches_df["rocket"]:

            if rocket_id not in rocket_dict:

                rocket_df = self.get_rocket(rocket_id)
                rocket_name.append(rocket_df["name"][0])
                rocket_dict[rocket_id] = rocket_df["name"][0]
            else:
                rocket_name.append(rocket_dict[rocket_id])

        return rocket_name

    def get_lauchpad_data_col(self, launches_df):
        launchpad_name = []
        launchpad_longitude = []
        launchpad_latitude = []

        launchpad_name_dict = {}
        launchpad_long_dict = {}
        launchpad_lat_dict = {}

        for launchpad_id in launches_df["launchpad"]:

            if launchpad_id not in launchpad_name_dict:

                launchpad_df = self.get_launchpad(launchpad_id)
                launchpad_name.append(launchpad_df["name"][0])
                launchpad_longitude.append(launchpad_df["longitude"][0])
                launchpad_latitude.append(launchpad_df["latitude"][0])

                launchpad_name_dict[launchpad_id] = launchpad_df["name"][0]
                launchpad_long_dict[launchpad_id] = launchpad_df["longitude"][0]
                launchpad_lat_dict[launchpad_id] = launchpad_df["latitude"][0]
            else:
                launchpad_name.append(launchpad_name_dict[launchpad_id])
                launchpad_longitude.append(launchpad_long_dict[launchpad_id])
                launchpad_latitude.append(launchpad_lat_dict[launchpad_id])

        return launchpad_name, launchpad_longitude, launchpad_latitude

    def get_payload_data_col(self, launches_df):
        launchpad_mass = []
        launchpad_orbit = []

        launchpad_mass_dict = {}
        launchpad_orbit_dict = {}

        for payload_id in launches_df["payloads"]:

            if payload_id not in launchpad_mass_dict:

                payload_df = self.get_payload(payload_id)
                launchpad_mass.append(payload_df["mass_kg"][0])
                launchpad_orbit.append(payload_df["orbit"][0])

                launchpad_mass_dict[payload_id] = payload_df["mass_kg"][0]
                launchpad_orbit_dict[payload_id] = payload_df["orbit"][0]

            else:
                launchpad_mass.append(launchpad_mass_dict[payload_id])
                launchpad_orbit.append(launchpad_orbit_dict[payload_id])

        return launchpad_mass, launchpad_orbit

    def get_cores_data_col(self, launches_df):
        core_blocks = []
        core_reuse_counts = []
        core_serials = []
        outcomes = []
        flights = []
        grid_fins = []
        reused = []
        legs = []
        landing_pads = []

        core_blocks_dict = {}
        core_reuse_counts_dict = {}
        core_serials_dict = {}

        for core in launches_df["cores"]:

            if core["core"] is not None:

                core_id = core["core"]

                if not core_id in core_blocks_dict:

                    core_df = self.get_core(core_id)
                    core_blocks.append(core_df["block"][0])
                    core_reuse_counts.append(core_df["reuse_count"][0])
                    core_serials.append(core_df["serial"][0])

                    core_blocks_dict[core_id] = core_df["block"][0]
                    core_reuse_counts_dict[core_id] = core_df["reuse_count"][0]
                    core_serials_dict[core_id] = core_df["serial"][0]
                else:
                    core_blocks.append(core_blocks_dict[core_id])
                    core_reuse_counts.append(core_reuse_counts_dict[core_id])
                    core_serials.append(core_serials_dict[core_id])

            else:
                core_blocks.append(None)
                core_reuse_counts.append(None)
                core_serials.append(None)

            outcomes.append(str(core['landing_success']) + ' ' + str(core['landing_type']))
            flights.append(core['flight'])
            grid_fins.append(core['gridfins'])
            reused.append(core['reused'])
            legs.append(core['legs'])
            landing_pads.append(core['landpad'])

        return core_blocks, core_reuse_counts, core_serials, outcomes, flights, grid_fins, reused, legs, landing_pads

    #endregion


    #region Public Methods

    def get_all_launches(self):
        launches_df = self.get_launches()

        launches_df = self.format_launches(launches_df)

        rocket_name_col = self.get_rocket_name_col(launches_df)

        lchpad_name_col, lchpad_longitude_col, lchpad_latitude_col = self.get_lauchpad_data_col(launches_df)

        payload_mass, payload_orbit = self.get_payload_data_col(launches_df)

        core_blocks, core_reuse_counts, core_serials, outcomes, flights, grid_fins, reused, legs, landing_pads = self.get_cores_data_col(
            launches_df)

        launch_dict = {'FlightNumber': list(launches_df['flight_number']),
                       'Date': list(launches_df['date']),
                       'BoosterVersion': rocket_name_col,
                       'PayloadMass': payload_mass,
                       'Orbit': payload_orbit,
                       'LaunchSite': lchpad_name_col,
                       'Outcome': outcomes,
                       'Flights': flights,
                       'GridFins': grid_fins,
                       'Reused': reused,
                       'Legs': legs,
                       'LandingPad': landing_pads,
                       'Block': core_blocks,
                       'ReusedCount': core_reuse_counts,
                       'Serial': core_serials,
                       'Longitude': lchpad_longitude_col,

                       'Latitude': lchpad_latitude_col}

        launch_pd = pd.DataFrame.from_dict(launch_dict)
        return launch_pd

    #endregion

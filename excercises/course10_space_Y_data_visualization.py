from typing import Optional

import pandas as pd
#from js import fetch
import io
import folium
# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon
from math import sin, cos, sqrt, atan2, radians

class Course10SpaceYDataVisualization:
    _CSV_PATH = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'


    def __init__(self):
        # Setting this option will print all collumns of a dataframe
        pd.set_option('display.max_columns', None)
        # Setting this option will print all of the data in a feature
        pd.set_option('display.max_colwidth', None)


    def show_johnson_space_center(self):
        # Start location is NASA Johnson Space Center
        nasa_coordinate = [29.559684888503615, -95.0830971930759]
        site_map = folium.Map(location=nasa_coordinate, zoom_start=10)

        # Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
        circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(
            folium.Popup('NASA Johnson Space Center'))
        # Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
        marker = folium.map.Marker(
            nasa_coordinate,

            # Create an icon as a text label
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(0, 0),
                html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
            )
        )
        site_map.add_child(circle)
        site_map.add_child(marker)

        site_map.show_in_browser()

    def get_mouse_position(self):
        formatter = "function(num) {return L.Util.formatNum(num, 5);};"
        mouse_position = MousePosition(
            position='topright',
            separator=' Long: ',
            empty_string='NaN',
            lng_first=False,
            num_digits=20,
            prefix='Lat:',
            lat_formatter=formatter,
            lng_formatter=formatter,
        )

        return  mouse_position

    def calculate_distance(self,lat1, lon1, lat2, lon2):
        # approximate radius of earth in km
        R = 6373.0

        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        return distance



    def show_launch_sites(self,launch_sites_df,launches_df):
        nasa_coordinate = [29.559684888503615, -95.0830971930759]
        site_map = folium.Map(location=nasa_coordinate, zoom_start=10)

        launch_site_coordinates=[]

        for index, row in launch_sites_df.iterrows():

            launch_site_coord=[row["Lat"],row["Long"]]
            launch_site_coordinates.append(launch_site_coord)
            circle = folium.Circle(launch_site_coord, radius=1000, color='#d35400', fill=True).add_child(
                folium.Popup(row["Launch Site"]))
            site_map.add_child(circle)

        marker_cluster = MarkerCluster()
        for index, launch in launches_df.iterrows():

            color="red" if launch["class"]==0 else "green"
            m_icon=DivIcon(icon_size=(20, 20),
                                           icon_anchor=(0, 0),
                                            html='<div style="font-size: 12; color:{};"><b>x</b></div>'.format(color))


            new_marker=folium.map.Marker([launch["Lat"], launch["Long"]], icon=m_icon)

            marker_cluster.add_child(new_marker)


        #- We draw a line to the nearest road
        road_coordinates = [28.55736, -80.58832]
        dist = self.calculate_distance(road_coordinates[0], road_coordinates[1], launch_site_coordinates[0][0],
                                       launch_site_coordinates[0][1])


        distance_marker = folium.Marker(
           road_coordinates,
           icon=DivIcon(
               icon_size=(20,20),
               icon_anchor=(0,0),
               html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(dist),
               )
           )

        line_coordinates = [
            road_coordinates,  # Road
            launch_site_coordinates[0]
        ]


        lines = folium.PolyLine(locations=line_coordinates, weight=1)
        site_map.add_child(lines)

        mouse_pos=self.get_mouse_position()

        site_map.add_child(mouse_pos)
        site_map.add_child(marker_cluster)
        site_map.add_child(distance_marker)

        site_map.show_in_browser()


    def download_and_display(self):
        launches_df = pd.read_csv(Course10SpaceYDataVisualization._CSV_PATH)

        launches_df = launches_df[['Launch Site', 'Lat', 'Long', 'class']]
        launch_sites_df = launches_df.groupby(['Launch Site'], as_index=False).first()
        launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]


        #self.show_johnson_space_center()
        self.show_launch_sites(launch_sites_df,launches_df)

        pass


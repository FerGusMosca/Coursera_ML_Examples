import pandas as pd
from dash import dash, callback
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input,Output
import plotly.express as px
from exams.exam10_presentation_calculations import Exam10PresentationCalculations


class Course10SpaceYDashboard:

    def __init__(self):
        # Setting this option will print all collumns of a dataframe
        pd.set_option('display.max_columns', None)
        # Setting this option will print all of the data in a feature
        pd.set_option('display.max_colwidth', None)

    def get_launches_from_csv(self):
        df = pd.read_csv("{}space_x_launches.csv".format(Exam10PresentationCalculations._INPUT_FOLDER))
        return df

    def prepare_combo_options(self):
        spacex_df = pd.read_csv("{}space_x_launches.csv".format(Exam10PresentationCalculations._INPUT_FOLDER))
        spacex_df=spacex_df.groupby("LaunchSite")['FlightNumber'].count().reset_index()

        launch_sites=[ {'label': 'All Sites', 'value': 'ALL'}]


        for launchSite in spacex_df["LaunchSite"]:
            launch_sites.append( {'label': launchSite, 'value': launchSite})

        return  launch_sites

    def init_dashboard(self):

        combo_options=self.prepare_combo_options()

        app = dash.Dash(title="Space X Launches Dashboard", suppress_callback_exceptions=True)
        app.layout = html.Div([
            html.H1("Space X Launches Dashboard",
                    style={"textAlign": "center", "color": "#503D36", "font-size": 24}),
            dcc.Dropdown(id='site_id',
                         options=combo_options,
                         value="ALL",
                         searchable=True,
                         placeholder='Select a Launch Site here',
                         style={"width": "80%", "padding ": "3px", "font-size": "20px", "text-align-last": "center"}
                         ),
            dcc.RangeSlider(id='payload-slider',
                            min=0,max=10000,step=1000,marks={0: '0',100: '100'},
                            value=[0, 10000]),

            html.Div(id='success-pie-chart', className="chart-grid", style={'display': 'flex'}),
            #html.Div(id='dd-output-container')
        ])

        app.run(debug=False)

    @staticmethod
    def get_success_launches(launches_df):
        #1- We filter the succesful launches
        launches_df = launches_df[(launches_df["Outcome"] == "True Ocean")
                                                    | (launches_df["Outcome"] == "True RTLS")
                                                    | (launches_df["Outcome"] == "True ASDS")]

        #2-xxx
        launches_df=launches_df.groupby("LaunchSite").count().reset_index().rename(columns={'FlightNumber': 'Count'})

        return launches_df[["LaunchSite","Count"]]


    @staticmethod
    def get_success_rate_by_launch_site(df_launches,launch_site):

        df_launches=df_launches[df_launches["LaunchSite"]==launch_site]

        succesful_landings_df = df_launches[(df_launches["Outcome"] == "True Ocean")
                                            | (df_launches["Outcome"] == "True RTLS")
                                            | (df_launches["Outcome"] == "True ASDS")]\
                                .groupby("LaunchSite").count().reset_index().rename(columns={'Outcome': 'Count'})[["LaunchSite","Count"]]
        succesful_landings_df["Status"]="Success"

        unsuccesful_landings_df = df_launches[(df_launches["Outcome"] != "True Ocean")
                                            & (df_launches["Outcome"] != "True RTLS")
                                            & (df_launches["Outcome"] != "True ASDS")] \
                                    .groupby("LaunchSite").count().reset_index().rename(columns={'Outcome': 'Count'})[["LaunchSite","Count"]]
        unsuccesful_landings_df["Status"] = "Failure"
        summary_df=pd.concat([succesful_landings_df, unsuccesful_landings_df])


        return summary_df


    # Function decorator to specify function input and output
    @callback(Output(component_id='success-pie-chart', component_property='children'),
              Input(component_id='site_id', component_property='value'))
    def get_pie_chart(entered_site):
        launches_df = pd.read_csv("{}space_x_launches.csv".format(Exam10PresentationCalculations._INPUT_FOLDER))
        title=""

        if entered_site == 'ALL':
            title="Total Success Launches By Site"
            launches_df=Course10SpaceYDashboard.get_success_launches(launches_df)
            R_chart = dcc.Graph(
                figure=px.pie(launches_df, names="LaunchSite", values="Count",
                              title=title))

            return R_chart


        else:
            title="Total Success Launches for {}".format(entered_site)
            launches_df=Course10SpaceYDashboard.get_success_rate_by_launch_site(launches_df,entered_site)

            R_chart = dcc.Graph(
                figure=px.pie(launches_df, names="Status", values="Count",
                              title=title))
            return R_chart
    # return the outcomes piechart for a selected site
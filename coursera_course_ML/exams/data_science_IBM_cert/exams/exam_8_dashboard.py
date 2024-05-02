from dash import dash, callback
import dash_html_components as html
import dash_core_components as dcc
from coursera_course_ML.util.file_downloader import FileDownloader
from dash.dependencies import Input,Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

class Module8Dashboard:
    download_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"
    output_path = "./input/automobiles_sales.csv"

    def __init__(self):
        FileDownloader.download(Module8Dashboard.download_url,Module8Dashboard.output_path)

    def init_dashboard(self):


        year_list = [i for i in range(1980, 2024, 1)]

        app=dash.Dash(title="Automobile Sales Statistics Dashboard" , suppress_callback_exceptions=True)
        app.layout = html.Div([
            html.H1("My Automobile Sales Statistics Dashboard",style={"textAlign": "center", "color": "#503D36", "font-size": 24}),
            dcc.Dropdown(id='dropdown-statistics',
                   options=[
                           {'label': 'Yearly Statistics', 'value': 'Yearly Statistics'},
                           {'label': 'Recession Period Statistics', 'value': 'Recession Period Statistics'}
                           ],
                  placeholder='Select a report type',
                  style={"width":"80%","padding ":"3px","font-size":"20px","text-align-last":"center"}
                         ),
            dcc.Dropdown(id='select-year',
                         options=[{'label': i, 'value': i} for i in year_list],
                         placeholder='Select a year',
                         style={"width": "80%", "padding ": "3px", "font-size": "20px", "text-align-last": "center"}
                         ),
            html.Div(id='output-container', className="chart-grid", style={'display': 'flex'}),
            html.Div(id='dd-output-container')
        ])



        app.run(debug=True)

    @callback(
        Output(component_id='select-year', component_property='disabled'),
        Input(component_id='dropdown-statistics', component_property='value'))
    def update_input_container(sel_value):
        if sel_value == 'Yearly Statistics':
            return False
        else:
            return True

    @staticmethod
    def get_line_avg_daily_sales_by_year(df,title):
        yearly_rec = df.groupby('Year')['Automobile_Sales'].mean().reset_index()
        R_chart1 = dcc.Graph(
            figure=px.line(yearly_rec,
                           x='Year',
                           y='Automobile_Sales',
                           title=title))
        return  R_chart1

    @staticmethod
    def get_line_sum_monthly_sales_by_month(df,title):
        monthly_df = df.groupby('Month')['Automobile_Sales'].sum().reset_index()
        R_chart2 = dcc.Graph(
            figure=px.line(monthly_df,
                           x='Month',
                           y='Automobile_Sales',
                           title=title))
        return R_chart2


    @staticmethod
    def get_bar_avg_vehicles_sold_by_vehicle_type(df,title):
        avg_by_type = df.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        R_chart2 = dcc.Graph(
            figure=px.bar(avg_by_type,
                           x='Vehicle_Type',
                           y='Automobile_Sales',
                           title=title))
        return R_chart2

    @staticmethod
    def get_bar_avg_daily_sales_by_year(df,title):
        yearly_rec = df.groupby('Year')['Automobile_Sales'].mean().reset_index()
        R_chart3= dcc.Graph(
            figure=px.bar(yearly_rec,
                           x='Year',
                           y='Automobile_Sales',
                           title=title))
        return R_chart3

    @staticmethod
    def get_pie_total_expenditure_by_vehicle_type(df,title):
        df["TotalExpenditure"]=df["Automobile_Sales"]*df["Price"]
        vech_type_rec = df.groupby('Vehicle_Type')['TotalExpenditure'].sum().reset_index()
        R_chart= dcc.Graph(
            figure=px.pie(vech_type_rec,names="Vehicle_Type",values="TotalExpenditure",
                           title=title))
        return R_chart

    @staticmethod
    def get_pie_total_advertising_expenditure_by_vehicle_type(df,title):

        vech_type_rec = df.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        R_chart= dcc.Graph(
            figure=px.pie(vech_type_rec,names="Vehicle_Type",values="Advertising_Expenditure",
                           title=title))
        return R_chart

    @staticmethod
    def get_bar_sales_by_vehicle_type_grp_unemployment(df):
        df_unempl_bin_1 = df[df["unemployment_rate"]<=3]#less than 3
        df_unempl_bin_2 = df[(df["unemployment_rate"] > 3) & (df["unemployment_rate"] <= 4)]#btw 3 and 4
        df_unempl_bin_3 = df[(df["unemployment_rate"] > 4) & (df["unemployment_rate"] <= 5)]#btw 4 and 5
        df_unempl_bin_4 = df[df["unemployment_rate"] > 5]  # more than 5

        df_unempl_bin_1 = df_unempl_bin_1.groupby('Vehicle_Type')['Automobile_Sales'].sum().reset_index()
        df_unempl_bin_2 = df_unempl_bin_2.groupby('Vehicle_Type')['Automobile_Sales'].sum().reset_index()
        df_unempl_bin_3 = df_unempl_bin_3.groupby('Vehicle_Type')['Automobile_Sales'].sum().reset_index()
        df_unempl_bin_4 = df_unempl_bin_4.groupby('Vehicle_Type')['Automobile_Sales'].sum().reset_index()


        fig = go.Figure(data=[
            go.Bar(name='Unempl, <=3%', x=df_unempl_bin_1["Vehicle_Type"], y=df_unempl_bin_1["Automobile_Sales"]),
            go.Bar(name='Unempl, 3%-4%', x=df_unempl_bin_2["Vehicle_Type"], y=df_unempl_bin_2["Automobile_Sales"]),
            go.Bar(name='Unempl, 4%-5%', x=df_unempl_bin_3["Vehicle_Type"], y=df_unempl_bin_3["Automobile_Sales"]),
            go.Bar(name='Unempl, >5%', x=df_unempl_bin_4["Vehicle_Type"],y=df_unempl_bin_4["Automobile_Sales"])

        ])

        R_chart = dcc.Graph(figure=fig)

        return R_chart


    @callback(
        Output(component_id='output-container', component_property='children'),
        [Input(component_id='dropdown-statistics', component_property='value'), Input(component_id='select-year', component_property='value')])
    def update_output_container(statistics_sel,year_sel):
        if statistics_sel == 'Recession Period Statistics':
            # Filter the data for recession periods
            df = pd.read_csv(Module8Dashboard.output_path)  # Path after downloading the file

            R_chart1 = Module8Dashboard.get_line_avg_daily_sales_by_year(df[df['Recession'] == 1],"Avg Daily Sales by Year on Recession")
            R_chart2 = Module8Dashboard.get_bar_avg_vehicles_sold_by_vehicle_type(df[df['Recession'] == 1],"Avg Daily Sales by Vehicle Type")
            R_chart3 = Module8Dashboard.get_pie_total_expenditure_by_vehicle_type(df[df['Recession'] == 1],"Total Expenditure by Vehicle Type on Recession")
            R_chart4 = Module8Dashboard.get_bar_sales_by_vehicle_type_grp_unemployment(df[df['Recession'] == 1])

            return [
                html.Div(className='chart-item', children=[html.Div(children=R_chart1)]),
                html.Div(className='chart-item', children=[html.Div(children=R_chart2)]),
                html.Div(className='chart-item', children=[html.Div(children=R_chart3)]),
                html.Div(className='chart-item', children=[html.Div(children=R_chart4)]),

            ]


        elif statistics_sel=="Yearly Statistics":
            df = pd.read_csv(Module8Dashboard.output_path)  # Path after downloading the file
            R_chart1 = Module8Dashboard.get_bar_avg_daily_sales_by_year(df,"Daily Sales by Year")
            R_chart2 = Module8Dashboard.get_line_sum_monthly_sales_by_month(df[df["Year"]==year_sel],"Total Sales by Month for {}".format(year_sel))
            R_chart3 = Module8Dashboard.get_bar_avg_vehicles_sold_by_vehicle_type(df[df["Year"]==year_sel],"Avg Daily Sales by Vehicle Type for {}".format(year_sel))
            R_chart4 = Module8Dashboard.get_pie_total_advertising_expenditure_by_vehicle_type(df[df["Year"] == year_sel],"Total Adv. Expenditure by Vehicle Type for {}".format(year_sel))

            return [
                html.Div(className='chart-item', children=[html.Div(children=R_chart1)]),
                html.Div(className='chart-item', children=[html.Div(children=R_chart2)]),
                html.Div(className='chart-item', children=[html.Div(children=R_chart3)]),
                html.Div(className='chart-item', children=[html.Div(children=R_chart4)])
            ]

    @callback(
        Output('dd-output-container', 'children'),
        Input('dropdown-statistics', 'value')
    )
    def update_output(value):
        return f'You have selected {value}'
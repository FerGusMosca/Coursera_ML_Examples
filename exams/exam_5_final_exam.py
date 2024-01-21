from dash import dash, callback
import dash_html_components as html
import dash_core_components as dcc
from util.file_downloader import FileDownloader
from dash.dependencies import Input,Output, State
from datetime import datetime
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import dash_table
_DATA_CSV = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/data/apple.json"


class Exam5FinalExam:


    def __init__(self):
        pass


    def init_dashboard(self):


        app=dash.Dash(title="Securities Dashboard" , suppress_callback_exceptions=True)
        app.layout = html.Div([
            html.H1("Securities Dashboard",style={"textAlign": "center", "color": "#503D36", "font-size": 24}),
            html.Label("Symbol: ",style={"width":"20%", "margin": "6px","padding ":"3px","font-size":"20px","text-align-last":"left"}),
            dcc.Textarea(id='txt-symbol',style={"width":"20%", "margin": "6px","verticalAlign":"middle","padding ":"3px","font-size":"20px","text-align-last":"left"}),
            dcc.DatePickerSingle(
                id='date-from',
                date=datetime(datetime.now().year, 1, 1),  # Initial default: January 1 of the current year,
                style={"margin": "6px"}
            ),
            dcc.DatePickerSingle(
                id='date-to',
                date=datetime.now(),  # Initial default: January 1 of the current year
                style={ "margin": "6px"}
            ),
            dcc.Dropdown(id='dropdown-statistics',
                   options=[
                           {'label': 'Stock Prices', 'value': 'Prices'},
                           {'label': 'Revenue', 'value': 'Revenue'}
                           ],
                  placeholder='Select a report type',
                  style={"width":"80%", "margin": "6px","padding ":"3px","font-size":"20px","text-align-last":"center"}
                         ),
            html.Button('Search', id='search-button',style={"font-size":"30px"}),
            html.Div(id='output-containet-data', className="chart-grid", style={'display': 'flex'}),
            html.Div(id='dd-output-container')
        ])



        app.run(debug=True)

    @callback(
        Output(component_id='output-containet-data', component_property='children'),
        [Input(component_id='search-button', component_property='n_clicks')],
        [State(component_id='dropdown-statistics', component_property='value'),
         State(component_id='txt-symbol', component_property='value'),
         State(component_id='date-from', component_property='date'),
         State(component_id='date-to', component_property='date')
         ]
    )
    def update_output_container(n_clicks,stats_sel,symbol,from_date,to_date):
        if stats_sel == 'Prices':
            start_date = from_date.split("T")[0]#We extract the minutes and seconds that are not important for Yahoo
            end_date = to_date.split("T")[0]#We extract the minutes and seconds that are not important for Yahoo

            tkr_mgr = yf.Ticker(symbol)
            symbol_prices_df = tkr_mgr.history(start=start_date, end=end_date)

            #we round all th columns
            numeric_columns = symbol_prices_df.select_dtypes(include='number').columns
            symbol_prices_df[numeric_columns] = symbol_prices_df[numeric_columns].round(2)

            #we format the date
            symbol_prices_df=symbol_prices_df.reset_index()
            # Format the date column to MM-dd-yyyy
            symbol_prices_df['Date'] = symbol_prices_df['Date'].dt.strftime('%m-%d-%Y')

            #we create the records dictionary
            data_dict=symbol_prices_df.to_dict('records')

            div_table = html.Div([
                dash_table.DataTable(
                    id='data-table',
                    columns=[{'name': col, 'id': col} for col in symbol_prices_df.columns],
                    data=data_dict,
                )
            ])

            return div_table


        elif stats_sel=="Revenue":
            pass
            # df = pd.read_csv(Module8Dashboard.output_path)  # Path after downloading the file
            # R_chart1 = Module8Dashboard.get_bar_avg_daily_sales_by_year(df,"Daily Sales by Year")
            # R_chart2 = Module8Dashboard.get_line_sum_monthly_sales_by_month(df[df["Year"]==year_sel],"Total Sales by Month for {}".format(year_sel))
            # R_chart3 = Module8Dashboard.get_bar_avg_vehicles_sold_by_vehicle_type(df[df["Year"]==year_sel],"Avg Daily Sales by Vehicle Type for {}".format(year_sel))
            # R_chart4 = Module8Dashboard.get_pie_total_advertising_expenditure_by_vehicle_type(df[df["Year"] == year_sel],"Total Adv. Expenditure by Vehicle Type for {}".format(year_sel))
            #
            # return [
            #     html.Div(className='chart-item', children=[html.Div(children=R_chart1)]),
            #     html.Div(className='chart-item', children=[html.Div(children=R_chart2)]),
            #     html.Div(className='chart-item', children=[html.Div(children=R_chart3)]),
            #     html.Div(className='chart-item', children=[html.Div(children=R_chart4)])
            # ]

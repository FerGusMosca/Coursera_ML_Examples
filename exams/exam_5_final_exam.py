import requests
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
import re
from bs4 import BeautifulSoup


_REV_LINK = "https://www.macrotrends.net/stocks/charts/{}/{}/revenue"
_TESTLA_REV_LIK="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm"
_GME_REV_LINK=" https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html"


class Exam5FinalExam:


    def __init__(self):
        pass
    @staticmethod
    def get_download_link(symbol,demo_mode):
        if not demo_mode:
            stock_name = Exam5FinalExam.split_by_multiple_characters(Exam5FinalExam.get_stock_name(symbol), [" ", ","])[0]
            return  _REV_LINK.format(symbol, stock_name.lower())
        else:
            if symbol=="TSLA":
                return _TESTLA_REV_LIK
            elif symbol=="GME":
                return  _GME_REV_LINK
            else:
                raise Exception("No pre cooked exception available in demo mode for symbol {}".format(symbol))


    @staticmethod
    def _get_url_html(symbol, demo_mode):
        url=Exam5FinalExam.get_download_link(symbol,demo_mode)

        if not demo_mode:

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
                  'Referer': url
            }

            response=None
            with requests.Session() as session:
                session.headers.update(headers)
                response = session.get(url,headers=headers, allow_redirects=True)

            return  response
        else:
            response=requests.get(url)
            return  response



    @staticmethod
    def split_by_multiple_characters(input_string, delimiters):
        # Use re.split with a regular expression that matches any of the specified delimiters
        pattern = '|'.join(map(re.escape, delimiters))
        result = re.split(pattern, input_string)
        return result

    @staticmethod
    def get_stock_name(symbol):
        try:
            stock_info = yf.Ticker(symbol)
            stock_name = stock_info.info.get('shortName', f"Name not available for {symbol}")
            return stock_name
        except Exception as e:
            raise  Exception ("No stock name found for symbol {}".format(symbol))

    @staticmethod
    def get_revenue(symbol):

        response=Exam5FinalExam._get_url_html(symbol,True)

        if response is not None and  response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            revenue_table = soup.find('table', {'class': 'historical_data_table table'})

            # Extract data from the revenue table
            revenue_data = []

            if revenue_table:
                rows = revenue_table.find_all('tr')
                for row in rows[1:]:  # Skipping the header row
                    columns = row.find_all('td')
                    date = columns[0].text.strip()
                    revenue = columns[1].text.strip().replace(',', '')  # Remove commas from revenue values
                    revenue_data.append({'Date': date, 'Revenue': revenue})

            # Convert to DataFrame
            revenue_df = pd.DataFrame(revenue_data)

            # Optionally, you can convert the 'Date' column to datetime format
            revenue_df['Date'] = pd.to_datetime(revenue_df['Date'])

            return revenue_df

        else:
            # Print an error message if the request was not successful
            raise Exception(f"Failed to fetch content. Status code: {response.status_code}")

    @staticmethod
    def get_prices_df(symbol,from_date,to_date):
        start_date = from_date.split("T")[0]  # We extract the minutes and seconds that are not important for Yahoo
        end_date = to_date.split("T")[0]  # We extract the minutes and seconds that are not important for Yahoo

        tkr_mgr = yf.Ticker(symbol)
        symbol_prices_df = tkr_mgr.history(start=start_date, end=end_date)

        # we round all th columns
        numeric_columns = symbol_prices_df.select_dtypes(include='number').columns
        symbol_prices_df[numeric_columns] = symbol_prices_df[numeric_columns].round(2)

        # we format the date
        symbol_prices_df = symbol_prices_df.reset_index()
        # Format the date column to MM-dd-yyyy
        symbol_prices_df['Date'] = symbol_prices_df['Date'].dt.strftime('%m-%d-%Y')

        return symbol_prices_df


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
                           {'label': 'Revenue', 'value': 'Revenue'},
                            {'label': 'Prices Graph', 'value': 'Prices_Graph'},
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
            symbol_prices_df=Exam5FinalExam.get_prices_df(symbol,from_date,to_date)

            #we create the records dictionary
            data_dict=symbol_prices_df.head().to_dict('records')

            div_table = html.Div([
                dash_table.DataTable(
                    id='data-table',
                    columns=[{'name': col, 'id': col} for col in symbol_prices_df.columns],
                    data=data_dict,
                )
            ])

            return div_table
        elif stats_sel=="Prices_Graph":
            symbol_prices_df=Exam5FinalExam.get_prices_df(symbol,from_date,to_date)

            figure = go.Figure(data=[go.Candlestick(x=symbol_prices_df['Date'],
                                                    open=symbol_prices_df['Open'],
                                                    high=symbol_prices_df['High'],
                                                    low=symbol_prices_df['Low'],
                                                    close=symbol_prices_df['Close'])])

            # Update layout
            figure.update_layout(title='Candlestick Chart for {}'.format(symbol),
                                 xaxis_title='Date',
                                 yaxis_title='Stock Price for '.format(symbol),
                                 xaxis_rangeslider_visible=False)

            R_chart = dcc.Graph(figure=figure)

            return R_chart


        elif stats_sel=="Revenue":
            revenue_df= Exam5FinalExam.get_revenue(symbol)

            revenue_df = revenue_df.sort_values(by='Date').tail()

            fig = px.bar(revenue_df, x='Date', y='Revenue', title='Revenue history for {}'.format(symbol),
                         labels={'Revenue': 'Revenue (millions)', 'Date': 'Date'})


            R_chart = dcc.Graph(figure=fig)

            return R_chart


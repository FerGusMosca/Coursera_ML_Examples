import seaborn as sns

from coursera_course_ML.util.file_downloader import FileDownloader
import pandas as pd
import matplotlib.pyplot as plt

class Module8Exam:
    download_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"
    output_path = "./input/automobiles_sales.csv"

    def __init__(self):
        FileDownloader.download(Module8Exam.download_url,Module8Exam.output_path)

    def task_1_1_create_sales_x_year_lineplot(self):
        df = pd.read_csv(Module8Exam.output_path)#Path after downloading the file
        df_sales_by_year = df.groupby(["Year"],as_index=False).sum()
        df_sales_by_year.plot(x='Year', y='Automobile_Sales')


    def task_1_2_sales_per_vehicle_type_lineplot(self):
        df = pd.read_csv(Module8Exam.output_path)  # Path after downloading the file
        df_sales_vehicle_type_by_year = df.groupby(["Year","Vehicle_Type"], as_index=False)["Automobile_Sales"].sum()

        df_sales_vehicle_type_by_year = df_sales_vehicle_type_by_year.groupby(['Vehicle_Type'])['Automobile_Sales']
        df_sales_vehicle_type_by_year.plot(kind='line')
        plt.xlabel('Vehicle Types')
        plt.ylabel('# Sales')
        plt.title('Sales Trend Vehicle-wise during Recession')
        plt.legend()
        plt.show()

    def task_1_3_sales_recession_no_recession_lineplot(self):
        df = pd.read_csv(Module8Exam.output_path)
        new_df = df.groupby('Recession')['Automobile_Sales'].mean().reset_index()
        # Create the bar chart using seaborn
        #plt.figure(figsize=(400,400))
        sns.barplot(x='Recession', y='Automobile_Sales', hue='Recession', data=new_df)
        plt.xlabel('Recession/No Recession')
        plt.ylabel('Sales')
        plt.title('Average Automobile Sales during Recession and Non-Recession')
        plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
        plt.show()


    def task_1_3_v2_sales_per_vehicle_type(self):
        df = pd.read_csv(Module8Exam.output_path)

        # Filter the data for recessionary periods
        recession_data = df[df['Recession'] == 1]

        dd = df.groupby(['Recession', 'Vehicle_Type'])['Automobile_Sales'].mean().reset_index()

        # Calculate the total sales volume by vehicle type during recessions
        # sales_by_vehicle_type = recession_data.groupby('Vehicle_Type')['Automobile_Sales'].sum().reset_index()

        # Create the grouped bar chart using seaborn
        #plt.figure(figsize=(10, 6))
        sns.barplot(x='Recession', y='Automobile_Sales', hue='Vehicle_Type', data=dd)
        plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
        plt.xlabel('Period')
        plt.ylabel('Average Sales')
        plt.title('Vehicle-Wise Sales during Recession and Non-Recession Period')

        plt.show()

    def task_1_4_sales_per_vehicle_type(self):
        df = pd.read_csv(Module8Exam.output_path)  # Path after downloading the file
        # Create dataframes for recession and non-recession period
        rec_data = df[df['Recession'] == 1]
        non_rec_data = df[df['Recession'] == 0]

        # Figure
        fig = plt.figure(figsize=(12, 6))

        # Create different axes for subploting
        ax0 = fig.add_subplot(1, 2, 1)  # add subplot 1 (1 row, 2 columns, first plot)
        ax1 = fig.add_subplot(1, 2, 2)  # add subplot 2 (1 row, 2 columns, second plot).

        # plt.subplot(1, 2, 1)
        sns.lineplot(x='Year', y='GDP', data=rec_data, label='Recession', ax=ax0)
        ax0.set_xlabel('Year')
        ax0.set_ylabel('GDP')
        ax0.set_title('GDP Variation during Recession Period')

        # # plt.subplot(1, 2, 2)
        sns.lineplot(x='Year', y='GDP', data=non_rec_data, label='Non Recession', ax=ax1)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('GDP')
        ax1.set_title('GDP')

        plt.tight_layout()
        plt.show()

    def task_1_5_sales_seasonality(self):
        df = pd.read_csv(Module8Exam.output_path)  # Path after downloading the file
        non_rec_data = df[df['Recession'] == 0]

        size = non_rec_data['Seasonality_Weight']  # for bubble effect

        sns.scatterplot(data=non_rec_data, x='Month', y='Automobile_Sales', size=size)

        # you can further include hue='Seasonality_Weight', legend=False)

        plt.xlabel('Month')
        plt.ylabel('Automobile_Sales')
        plt.title('Seasonality impact on Automobile Sales')

        plt.show()

    def task_1_6_consumer_conf_to_sales(self):
        df = pd.read_csv(Module8Exam.output_path)  # Path after downloading the file
        # Create dataframes for recession and non-recession period
        rec_data = df[df['Recession'] == 1]
        plt.scatter(rec_data['Consumer_Confidence'], rec_data['Automobile_Sales'])

        plt.xlabel('Consumer Conf.')
        plt.ylabel('Sales')
        plt.title('Regr. Confidence/Sales')
        plt.show()

    def task_1_6_avg_px_sales_volume(self):
        df = pd.read_csv(Module8Exam.output_path)  # Path after downloading the file
        # Create dataframes for recession and non-recession period
        rec_data = df[df['Recession'] == 1]
        plt.scatter(rec_data['Price'], rec_data['Automobile_Sales'])

        plt.xlabel('Price')
        plt.ylabel('# Sales')
        plt.title('Price/Sales')
        plt.show()


    def task_1_7_recession_non_recession_advertising(self):
        df = pd.read_csv(Module8Exam.output_path)  # Path after downloading the file
        # Filter the data
        Rdata = df[df['Recession'] == 1]
        NRdata = df[df['Recession'] == 0]

        # Calculate the total advertising expenditure for both periods
        RAtotal = Rdata['Advertising_Expenditure'].sum()
        NRAtotal = NRdata['Advertising_Expenditure'].sum()

        # Create a pie chart for the advertising expenditure
        plt.figure(figsize=(8, 6))

        labels = ['Recession', 'Non-Recession']
        sizes = [RAtotal, NRAtotal]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

        plt.title('Advertising Expenditure during Recession and Non-Recession Periods')

        plt.show()

    def task_1_8_adv_expenditures_during_recessioln(self):
        df = pd.read_csv(Module8Exam.output_path)  # Path after downloading the file
        # Filter the data
        Rdata = df[df['Recession'] == 1]

        df_exp_by_type = Rdata.groupby("Vehicle_Type")['Advertising_Expenditure'].sum()

        plt.figure(figsize=(8, 6))

        labels = df_exp_by_type.index
        sizes = df_exp_by_type.values
        plt.pie(sizes,labels=labels,  autopct='%1.1f%%', startangle=90)

        plt.title('Share of Each Vehicle Type in Total Sales during Recessions')

        plt.show()

    def task_1_9_unempl_and_vehicle_type_sales(self):
        df = pd.read_csv(Module8Exam.output_path)  # Path after downloading the file
        data = df[df['Recession'] == 1]

        plt.figure(figsize=(10, 6))

        sns.countplot(data=data, x='unemployment_rate', hue='Vehicle_Type')

        plt.xlabel('Unemployment Rate')
        plt.ylabel('Count')
        plt.title('Effect of Unemployment Rate on Vehicle Type and Sales')
        plt.legend(loc='upper right')
        plt.show()


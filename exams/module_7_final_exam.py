import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from util.file_downloader import FileDownloader
import pandas as pd

class Module7Exam:
    download_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv"
    output_path = "./input/kc_house_data_NaN.csv"

    def __init__(self):
        FileDownloader.download(Module7Exam.download_url,Module7Exam.output_path)



    def q1_display_data_types(self):
        df = pd.read_csv(Module7Exam.output_path)#Path after downloading the file

        print ("Head : {}".format(df.head()))

        print("dTypes: {}".format(df.dtypes))

        print("Describe : {}".format(df.describe()))
        return df

    def q2_drop_and_describe(self,df):
        df=df.drop(columns=["id","Unnamed: 0"])
        print("Describe after drop:{}".format(df.describe()))

    def q2_extra_analysis(self,df):
        df = df.drop(columns=["id", "Unnamed: 0"])

        print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
        print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

        #replace missing values of bedrooms/bathrooms with the mean value
        mean = df['bedrooms'].mean()
        df['bedrooms'].replace(np.nan, mean, inplace=True)

        mean = df['bathrooms'].mean()
        df['bathrooms'].replace(np.nan, mean, inplace=True)


        #now we do not have more NaN values on these columns
        print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
        print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
        return df

    def q3_count_unique_floor_values(self,df):
        df_distinct_floor= df["floors"].value_counts().to_frame()

        print("distinct floor values:{}".format(df_distinct_floor))

    def q4_boxplot_eval_outliers(self,df):
        sns.boxplot(x="waterfront",y="price",data=df)

    def q5_regplot_eval(self,df):
        sns.regplot(x="sqft_above",y="price",data=df)

        print("Most correlated feature: {}".format(df.corr()['price'].sort_values()))

    def q6_fit_linear_regression_model(self,df):
        X = df[['sqft_above']]
        Y = df['price']
        lm = LinearRegression()
        lm.fit(X, Y)
        R2=lm.score(X, Y)

        sns.regplot(x="sqft_above", y="price", data=df)

        print("R2 for the trained model is {}".format(R2))

    def q6_bis_fit_linear_regression_model_only_R2(self, df):
            X = df[['sqft_living']]
            Y = df['price']
            lm = LinearRegression()
            lm.fit(X, Y)
            R2 = lm.score(X, Y)
            print("R2 for the trained model is {}".format(R2))

        #yhat=lm.predict(X)

    def q7_fit_multilinear_regr_model(self,df):
        features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15","sqft_above", "grade", "sqft_living"]

        #we replace the missing values w/the mean
        mean = df['bedrooms'].mean()
        df['bedrooms'].replace(np.nan, mean, inplace=True)

        mean = df['bathrooms'].mean()
        df['bathrooms'].replace(np.nan, mean, inplace=True)


        Z=df["sqft_living"]
        Y = df['price']
        lm = LinearRegression()
        lm.fit(Z, Y)
        R2 = lm.score(Z, Y)
        print("R2 for the trained model is {}".format(R2))

    def q8_pipeline_evaluation(self,df):
        features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15","sqft_above", "grade", "sqft_living"]

        # we replace the missing values w/the mean
        mean = df['bedrooms'].mean()
        df['bedrooms'].replace(np.nan, mean, inplace=True)

        mean = df['bathrooms'].mean()
        df['bathrooms'].replace(np.nan, mean, inplace=True)

        input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]

        pipe=Pipeline(input)

        Z=df[features]
        pipe.fit(Z,df["price"])
        R2 = pipe.score(Z, df["price"])

        print("R2 for the trained pipeline is {}".format(R2))

    def q9_ridge_eval_model(self,df):
        features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15",
                    "sqft_above", "grade", "sqft_living"]

        #we replace the missing values w/the mean
        mean = df['bedrooms'].mean()
        df['bedrooms'].replace(np.nan, mean, inplace=True)

        mean = df['bathrooms'].mean()
        df['bathrooms'].replace(np.nan, mean, inplace=True)

        X = df[features]
        Y = df['price']

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

        print("number of test samples:", x_test.shape[0])
        print("number of training samples:", x_train.shape[0])

        rmodel=Ridge(alpha=0.1)
        rmodel.fit(x_train,y_train)

        y_pred=rmodel.predict(x_test)

        R2 = rmodel.score(x_test, y_test)

        print("R2 for the Ridge model is {}".format(R2))

    def q10_ridge_w_plynomial_transform(self, df):
        features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15",
                    "sqft_above", "grade", "sqft_living"]

        # we replace the missing values w/the mean
        mean = df['bedrooms'].mean()
        df['bedrooms'].replace(np.nan, mean, inplace=True)

        mean = df['bathrooms'].mean()
        df['bathrooms'].replace(np.nan, mean, inplace=True)

        X = df[features]
        Y = df['price']

        pr=PolynomialFeatures(degree=2)

        X_transf=pr.fit_transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X_transf, Y, test_size=0.15, random_state=1)

        rmodel=Ridge(alpha=0.1)
        rmodel.fit(x_train,y_train)

        R2 = rmodel.score(x_test, y_test)

        print("R2 for the Ridge model (X Transformed) is {}".format(R2))






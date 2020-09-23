#this program explores the data set of the housing.csv.

import pandas as pd

data = pd.read_csv("housing.csv")



pd.set_option('display.max_columns', None)
print(data.head())


print(data.info())
print(data["ocean_proximity"].value_counts())


print(data.describe())



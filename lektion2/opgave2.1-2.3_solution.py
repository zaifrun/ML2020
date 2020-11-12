

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv("housing.csv")
print(data.info())

print("********** Cleaning data************")
for (columnName, columnData) in data.iteritems():
   print('Colunm Name : ', columnName)
   if data[columnName].isnull:
       print(columnName+" has empty or nan values")

#cleaning data - crude way of dropping any rows with null data
print("************ after dropping data*************")
data.dropna(inplace=True)
print(data.info())



pd.set_option('display.max_columns', None)
print(data.head())


print(data.info())
print(data["ocean_proximity"].value_counts())


print(data.describe())

#antal samples = 20640
#antal features = 10 (10 kolonner)

#hus toppe - ca: 16, 36, 52 år
# Så der har været nogle bygge booms.


data.hist(bins=50,figsize=(20,15))


corr_matrix = data.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

data.plot(kind = "scatter", x = "total_rooms", y = "median_house_value")

data.plot(kind= "scatter", x = "median_income", y = "median_house_value")


print("samples in data data: "+str(len(data)))


train_set, test_set = train_test_split(data,test_size=0.2,random_state=42)
print("samples in training set: "+str(len(train_set)))
print("sammples in test set: "+str(len(test_set)))


plt.figure()
data.boxplot(column =['median_house_value'], grid = False)
data.boxplot(by ='ocean_proximity', column =['median_house_value'], grid = False)

print("************ encoding data*************")

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
encoded_ocean_proximity = ordinal_encoder.fit_transform(data[["ocean_proximity"]])
data["ocean_proximity"] = encoded_ocean_proximity
print(data.describe())

print("********* Scaling data **********")

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
df_scaled = pd.DataFrame(scaled_data)
print(df_scaled.describe())

plt.show()



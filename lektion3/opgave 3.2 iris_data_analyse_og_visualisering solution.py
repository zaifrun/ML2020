
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("iris.csv")



pd.set_option('display.max_columns', None)
print(data.head())
print("Basic statistics")
print(data.describe())



data.hist(bins=50,figsize=(20,15))

print("correlations")
corr_matrix = data.corr()
print(corr_matrix)

data.plot(kind = "scatter", x = "petal.width", y = "petal.length")

plt.show()
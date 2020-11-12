#Opgave 1.3 - plots etc

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Du kan finde dokumentation for pyplot her: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.html


X = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
Y = np.array([550,710,730,710,685,740,880,1155,1207,1425,1575,1750,1890])


plt.plot(X,Y, "b.")  # hvad betyder "b." ? Se dokumentationen for plot i pyplot linket ovenover
#plt.axis([0,2,0,15])  # betyder parameterne her?

#training the model
lin_reg = LinearRegression()
lin_reg.fit(X.reshape(-1,1),Y)   # train the model on the data

#calculating the score
score = lin_reg.score(X.reshape(-1,1), Y)
print("score "+str(score))

#get parameters
a = lin_reg.coef_[0]
b = lin_reg.intercept_
print(" a = "+str(a))
print(" b = "+str(b))
X_predict = np.array([15])  # put the dates of which you want to predict kwh here

y_predict = lin_reg.predict(X_predict.reshape(-1,1))
print("predicted value for 15 (28/10) = "+str(y_predict))


# y = a * x + b)
# dvs x = (y-b) / a

x_3500 = (3500-b) / a

print("x værdi for y = 3500 gram i uge nummer "+str(x_3500))
# bliver i uge 29 - dvs. 3. februar 2021...

#plot the best line.
line_x = np.linspace(0,15,100)
line_y = a * line_x + b
plt.plot(line_x, line_y, '-r', label='best line')


plt.plot()
plt.show()



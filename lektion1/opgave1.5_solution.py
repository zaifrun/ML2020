#Opgave 1.3 - plots etc

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

# Du kan finde dokumentation for pyplot her: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.html

X = 2 * np.random.rand(100, 1)  # hvad betyder de to parametre 100 og 1?
y = 4 + 3 * X + np.random.randn(100, 1) # så hvilke parametre ville vi forvente modellen har?
#hvad er forskellen mellem rand og randn? (du må se om du kan finde dokumentationen selv....)


# Adding data.
#X = np.append(X,[[10]],axis=0)
#y = np.append(y,[[5]],axis=0)
#print(X)
#print(y)

plt.plot(X,y, "b.")  # hvad betyder "b." ? Se dokumentationen for plot i pyplot linket ovenover
plt.axis([0,2,0,15])  # betyder parameterne her?
plt.plot()

#training the model
lin_reg = LinearRegression()
lin_reg.fit(X,y)   # train the model on the data

#calculating the score
score = lin_reg.score(X,y)
print("score "+str(score))

#using the model on new data
new_x = np.array([[0],[2]])
y_predict = lin_reg.predict(new_x)

#plot the new data
plt.plot(new_x,y_predict,"r")


#how to get a and b parameters - i.e. the trained parameters for the model...?

print(" a = "+str(lin_reg.coef_))
print(" b = "+str(lin_reg.intercept_))

#train the model using SGD
sgd_reg = SGDRegressor(max_iter=10, tol=1e-3,penalty=None,eta0=0.1)
sgd_reg.fit(X,y.ravel())

print(" a (sgd) = "+str(sgd_reg.coef_))
print(" b (sgd) = "+str(sgd_reg.intercept_))
print(" score (sgd)"+str(sgd_reg.score(X,y)))


plt.show()






#An attempt at a solution to the last exercise in Machine Learning - Week 5 - Neural Networks 2.
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


import matplotlib.pyplot as plt


data=pd.read_csv('titanic_train_500_age_passengerclass.csv', sep=',', header=0)
# show the data
print(data.describe(include='all'))
print(data.values)

x = data["Age"]
y = data["Pclass"]

plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)


yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
yvalues["Survived"] = data["Survived"].copy()

ytrain = yvalues.head(400)
ytest = yvalues.tail(100)

# we have copied that to the y arrays, so we should drop it from the input
data.drop('Survived', axis=1, inplace=True)
data.drop('PassengerId',axis=1,inplace=True)


xtrain = data.head(400).copy()
xtest = data.tail(100).copy()

#replace missing values in xtrain with average (of xtrain age)
avg = xtrain['Age'].mean()
xtrain['Age'].fillna(avg,inplace=True)

#replace missing values in xtest with average (of xtest age)
avg = xtest['Age'].mean()
xtest['Age'].fillna(avg,inplace=True)


print(ytrain.describe(include='all'))
print(ytest.describe(include='all'))
print(xtrain.describe(include='all'))
print(xtest.describe(include='all'))

survived_training = ytrain['Survived'].sum()
survived_test = ytest['Survived'].sum()
print("Survived in the training set : "+str(survived_training))
print("Survived in the test set : "+str(survived_test))

scaler = StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest= scaler.transform(xtest)



mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000,random_state=42)

#mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000,batch_size=32,learning_rate_init=0.01,random_state=0)


mlp.fit(xtrain,ytrain.values.ravel())



predictions = mlp.predict(xtest)
print("PREDICTIONS")
print(predictions)

matrix = confusion_matrix(ytest, predictions)
print(matrix)
print(classification_report(ytest, predictions))

#By definition a confusion matrix C is such that C_{i, j} is equal to the number
# of observations known to be in group i but predicted to be in group j.

#Thus in binary classification, the count of true negatives is C_{0,0},
# false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.

tn, fp, fn, tp = matrix.ravel()
print("(TN,FP,FN,TP)",(tn, fp, fn, tp))
print("TN : 0 in test set and predicted as 0")
print("FP : 0 in test set and predicted FALSELY as 1")
print("FN : 1 in test set and predicted FALSELY as 0")
print("TP : 1 in test set and predicted as 1")
summen = tp + tn + fp + fn
pred_correct = (tp+tn)/summen
print("percentage predicted correctly: "+str(pred_correct))

plt.show()

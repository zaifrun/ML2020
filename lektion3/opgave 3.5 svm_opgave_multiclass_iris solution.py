
from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

iris = load_iris()  # load iris sample dataset
print(iris)
X = iris.data
y = iris.target

# check how many samples we have
print("Number of samples: " +str(len(y)))
print("X values")
print(X)
print("Y values")
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("samples in X training set: "+str(len(X_train)))
print("sammples in test set: "+str(len(X_test)))

svm = LinearSVC(C=1,loss="hinge",max_iter=5000)
svm.fit(X_train,y_train.ravel())
predictions = svm.predict(X_test)

print(predictions)


matrix = confusion_matrix(y_test,predictions) #compare
print(matrix)
#output is the confusion matrix
#tn, fp, fn, tp = matrix.ravel()
#print("(TN,FP,FN,TP)",(tn, fp, fn, tp))

print(classification_report(y_test,predictions))


print("precision: "+ str(precision_score(y_test,predictions,average='weighted')))
print("recall: "+ str(recall_score(y_test,predictions,average='weighted')))

print("f1 score "+ str(f1_score(y_test,predictions,average='weighted')))

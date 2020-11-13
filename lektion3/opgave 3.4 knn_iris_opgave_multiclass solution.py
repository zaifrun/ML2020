"""
This is used for the knn iris excercise -
whole dataset, multi-class predictions using KNN

"""
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

iris = load_iris()  # load iris sample dataset
X = iris.data # petal length and width, so 2D information
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("samples in X training set: "+str(len(X_train)))
print("sammples in test set: "+str(len(X_test)))
print(X)
print(y)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
knn.fit(X_train,y_train)

#Predict Output
predictions= knn.predict(X_test)

print("Actual")
print(y_test)
print("predictions:")
print(predictions)

print("Evaluating performance: Confusion matrix")
matrix = confusion_matrix(y_test,predictions) #compare
print(matrix)
#output is the confusion matrix

print(classification_report(y_test,predictions))

print("precision: "+ str(precision_score(y_test,predictions,average='weighted')))
print("recall: "+ str(recall_score(y_test,predictions,average='weighted')))
print("f1 score "+ str(f1_score(y_test,predictions,average='weighted')))
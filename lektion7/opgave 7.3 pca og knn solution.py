
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()  # load iris sample dataset
#print(iris)
X = iris.data # petal length and width, so 2D information
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("samples in X training set: "+str(len(X_train)))
print("sammples in test set: "+str(len(X_test)))

pca = PCA(n_components=2)
print("dimensions of original data training data")
print(X_train.shape)

pca.fit(X_train)
X_train_reduced = pca.transform(X_train)
print("dimensions of transformed training data")
print(X_train_reduced.shape)
X_test_reduced = pca.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

def doKnn(X_train,y_train,X_test,y_test):
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict Output
    predictions = knn.predict(X_test)

    print("Evaluating performance: Confusion matrix")
    matrix = confusion_matrix(y_test, predictions)  # compare
    print(matrix)
    # output is the confusion matrix

    print(classification_report(y_test, predictions))

    print("precision: " + str(precision_score(y_test, predictions, average='weighted')))
    print("recall: " + str(recall_score(y_test, predictions, average='weighted')))

    print("f1 score " + str(f1_score(y_test, predictions, average='weighted')))


print("******** KNN på pca PCA reduceret test **************")
doKnn(X_train_reduced,y_train,X_test_reduced,y_test)
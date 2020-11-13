# KNN på MNIST data
# multiclass classification

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score

print("MNIST example")
print("fetching data.....can take some time....")

from mlxtend.data import loadlocal_mnist

X_train, y_train = loadlocal_mnist(
            images_path='train-images.idx3-ubyte',
            labels_path='train-labels.idx1-ubyte')

X_test, y_test = loadlocal_mnist(
            images_path='t10k-images.idx3-ubyte',
            labels_path='t10k-labels.idx1-ubyte')

print("have data now")



print(X_train.shape)
print(y_train.shape)


X_test = X_test[0:1000] # Så får vi kun de første 1000
y_test = y_test[0:1000] # Tilsvarende for labels.


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

print("Training the model... please wait..")
knn.fit(X_train, y_train)
print("Model is trained - making predictions")
predictions = knn.predict(X_test)
print("Predictions done - now making classification report")

# printing confusion matrix
matrix = confusion_matrix(y_test,predictions) #compare
print(matrix)

# looking at how good our predictor is.
print(classification_report(y_test,predictions))

print("precision: "+ str(precision_score(y_test,predictions,average='weighted')))
print("recall: "+ str(recall_score(y_test,predictions,average='weighted')))

print("F1 score: "+str(f1_score(y_test,predictions,average='weighted')))


plt.show()
print("end of program")
# example of using mnist set for classification using logistic regression
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score

"""
print("MNIST example")
print("fetching data.....can take some time....")
#note fetching can take 1-2 minutes if not cached before
mnist = fetch_openml("mnist_784", version = 1, cache=True)


print("have data now")

mnist.keys()
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)
#convert the string array to int array
y = y.astype(np.uint8)
"""
plt.figure(1)
plt.axis("on")

"""
X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]
"""


from mlxtend.data import loadlocal_mnist

X_train, y_train = loadlocal_mnist(
            images_path='train-images.idx3-ubyte',
            labels_path='train-labels.idx1-ubyte')

X_test, y_test = loadlocal_mnist(
            images_path='t10k-images.idx3-ubyte',
            labels_path='t10k-labels.idx1-ubyte')

# visualize the distributions
plt.hist(y_train,bins=10)

#convert this into a binary classification problem
y_train_5 = (y_train ==5) # creates an array of true if 5, false otherwise
y_test_5 = (y_test == 5) # creates an array of true if 5, false otherwise

clf = LogisticRegression(random_state=42,max_iter=5000)
print("Training the model... please wait..")
clf.fit(X_train, y_train_5)
print("Model is trained")
predictions = clf.predict(X_test)


print("Evaluating performance: Confusion matrix")
matrix = confusion_matrix(y_test_5,predictions)
print(matrix)
#output is the confusion matrix

tn, fp, fn, tp = matrix.ravel()
print("(TN,FP,FN,TP)",(tn, fp, fn, tp))
print("precision: "+ str(precision_score(y_test_5,predictions)))
print("recall: "+ str(recall_score(y_test_5,predictions)))

print("F1 score: "+str(f1_score(y_test_5,predictions)))

print(classification_report(y_test_5,predictions))


plt.show()
print("end of program")
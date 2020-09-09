# example of using mnist set for classification
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier


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

some_digit = X[0]
print("label : "+str(y[0]))


some_digit_image = some_digit.reshape(28,28)
plt.figure(0)

plt.imshow(some_digit_image, cmap = "binary")
plt.axis("off")
plt.figure(1)
plt.axis("on")

X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]

plt.hist(y,bins=10)

y_train_5 = (y_train ==5) # creates an array of true if 5, false otherwise
y_test_5 = (y_test == 5) # creates an array of true if 5, false otherwise

sgd_clf = SGDClassifier(random_state=42)
print("Training the model... please wait..")
sgd_clf.fit(X_train, y_train_5)
print("Model is trained")

print("digit is five : "+str(sgd_clf.predict([some_digit])))



plt.show()
print("end of program")
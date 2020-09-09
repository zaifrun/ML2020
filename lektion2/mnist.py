# example of using mnist set for classification
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt


print("MNIST example")
print("fetching data.....can take some time....")
#note fetching can take 1-2 minutes if not cached before
mnist = fetch_openml("mnist_784", version = 1)


print("have data now")

mnist.keys()
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

some_digit = X[0]
print("label : "+y[0])


some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = "binary")
plt.axis("off")

X_train = X[60000:]
X_test = X[:60000]
y_train = y[60000:]
y_test = y[:60000]



plt.hist(y, normed=True, bins=10)

plt.show()


print("end of program")
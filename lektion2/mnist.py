# example of using mnist set for classification
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt


print("MNIST example")
print("fetching data")
#note fetching can take 1-2 minutes if not cached before
mnist = fetch_openml("mnist_784", version = 1)

#mndata = MNIST('./')
#images, labels = mndata.load_training()

print("have data now")


mnist.keys()
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

some_digit = X[0]

some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = "binary")
plt.axis("off")
print("label : "+y[0])

plt.show()





print("end of program")
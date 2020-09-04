print ("hello world")
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
data = np.random.randn(2, 100)

fig, axs = plt.subplots(3, 2, figsize=(6, 6))
axs[0, 0].hist(data[0])
axs[1, 0].scatter(data[0], data[1])
axs[0, 1].plot(data[0], data[1])
axs[1, 1].hist2d(data[0], data[1])
x = [1,2,3,4,5]
y = [2,4,6,8,10]
axs[2,0].scatter(x,y)
plt.show()

#Opgave 1.3 - plots etc

import matplotlib.pyplot as plt
import numpy as np

# Du kan finde dokumentation for pyplot her: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.html

X = 2 * np.random.rand(100, 1)  # hvad betyder de to parametre 100 og 1?
y = 4 + 3 * X + np.random.randn(100, 1)
#hvad er forskellen mellem rand og randn? (du må se om du kan finde dokumentationen selv....)

plt.plot(X,y, "b.")  # hvad betyder "b." ? Se dokumentationen for plot i pyplot linket ovenover
plt.axis([0,2,0,15])  # betyder parameterne her?
plt.plot()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans # This will be used for the algorithm
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)

fig, ax = plt.subplots()
plt.title('The raw data points - before clustering - we know the result!')
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

k = 2
#running kmeans clustering into two
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
# this will contain the labels for our predicted clusters (either 0 or 1)
labels = kmeans.labels_
# the centers of the calculated clusters
clusters = kmeans.cluster_centers_
# printing our cluster centers - there will be 2 of them.
print(clusters)


plt.figure()
plt.title('Clustering - by Kmeans - k = 2')
cmap = ListedColormap(['#FF0000', '#00FF00'])

plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolor='black', cmap=cmap, s=20)
plt.plot(clusters[0],clusters[1],'ys',markersize=15)



plt.show()

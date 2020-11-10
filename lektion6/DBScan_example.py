import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

#NOTE: Most of the code here is taken from a scikit-learn example.

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]  # we want 3 centers
#make some random data, 750 data points around the centers, with a standard deviation size of 0.4
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X) #scaling the values

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
#initialize an array of zeros (i.e. false) - the array has an index for every point
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#assign true to the all the samles indicated as core_points by the algorithm
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_  # the array of labels assigned to the points - i.e. cluster numbers

# Number of clusters in labels, ignoring noise if present
# Noise is also assigned a label (-1), all the real clusters are assigned
# values from 0,1,2...and so on. So if there is noise, then subtract 1 from the size of
# the labels to get the number of clusters.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1) # count the number of points that is labelled noise

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#various performance scores - you can look them up the in the documentation for
#DB scan
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plotting the  result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)

print(X)


fig, ax = plt.subplots()
scale = 5
ax.scatter(X[:,0], X[:,1], c='b', marker='o', s=scale,
               alpha=0.9, edgecolors='none')

ax.legend()
ax.grid(True)

plt.figure()
#make some colors along a gradient from 0 to 1 in the colorspace and we need the number of labels = number
#of colors
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
#loop over all the labels (1 for every point) and also map a label to a color at the same time.
for k, col in zip(unique_labels, colors):
    # is it a noise label? (-1)
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]


    class_member_mask = (labels == k)

    # the core points have a different size - the are bigger
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    # the non-core points will be smaller in size - but they still belong to the cluster of course
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
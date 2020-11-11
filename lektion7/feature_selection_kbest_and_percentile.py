
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile

iris = load_iris()  # load iris sample dataset
X = iris.data
y = iris.target

#print(iris.feature_names)
#print(X.shape)
selector = SelectKBest(k=2)
X_new = selector.fit_transform(X, y)
print(X_new.shape)

features_selected = selector.get_support(indices=True)
print(features_selected)
for i in features_selected:
    print(iris.feature_names[i])



#print(iris.feature_names)
#print(X.shape)
selector = SelectPercentile(percentile=75)
X_new = selector.fit_transform(X, y)
print(X_new.shape)

features_selected = selector.get_support(indices=True)
print(features_selected)
for i in features_selected:
    print(iris.feature_names[i])
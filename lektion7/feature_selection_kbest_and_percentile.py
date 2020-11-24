
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile

print("*********** selectKbest ************\n")


iris = load_iris()  # load iris sample dataset
X = iris.data
y = iris.target
print("Original dimensions of data:")
print(X.shape)

selector = SelectKBest(k=2)

print("transformed dimensions of data:")
X_new = selector.fit_transform(X, y)
print(X_new.shape)

features_selected = selector.get_support(indices=True)

print("scores of features:")
print(selector.scores_)
print("features selected (indexing starts at 0):")
print(features_selected)

print("labels of features selected:")
for i in features_selected:
    print(iris.feature_names[i])

print("*********** select percentile - 75% ************ \n")

selector = SelectPercentile(percentile=75)
X_new = selector.fit_transform(X, y)

print("Original dimensions of data:")
print(X.shape)
print("Transformed dimensions of data:")
print(X_new.shape)

features_selected = selector.get_support(indices=True)
print("Features selected:")
print(features_selected)
print("Feature names selected:")
for i in features_selected:
    print(iris.feature_names[i])
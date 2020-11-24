import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import pydotplus
import matplotlib.image as mpimg
import io

X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
print("size of x_train = "+str(len(X_train)))
print("size of y_train = "+str(len(y_train)))
print("size of x_test = "+str(len(X_test)))
print("size of y_test = "+str(len(y_test)))


tree_clf = DecisionTreeClassifier(max_depth=6)
tree_clf.fit(X,y) # training the classifier

predictions = tree_clf.predict(X_test)

print("Evaluating performance: Confusion matrix")
matrix = confusion_matrix(y_test,predictions)
print(matrix)
#output is the confusion matrix

tn, fp, fn, tp = matrix.ravel()
print("(TN,FP,FN,TP)",(tn, fp, fn, tp))
print("precision: "+ str(precision_score(y_test,predictions)))
print("recall: "+ str(recall_score(y_test,predictions)))

print("F1 score: "+str(f1_score(y_test,predictions)))

print(classification_report(y_test,predictions))


dot_data = io.StringIO()
target_names = ["0","1"]

export_graphviz(tree_clf,
                out_file=dot_data, # or put a filename here filename like "graph.dot", you then need to convert it into pgn
                rounded=True,
                class_names=target_names,
                filled=True)

filename = "tree.png"
pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png(filename) # write the dot data to a pgn file
img=mpimg.imread(filename) # read this pgn file

plt.figure(figsize=(8,8)) # setting the size to 10 x 10 inches of the figure.
imgplot = plt.imshow(img) # plot the image.

# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()
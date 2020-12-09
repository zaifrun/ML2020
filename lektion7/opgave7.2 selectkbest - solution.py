

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()  # load iris sample dataset
#print(iris)
X = iris.data # petal length and width, so 2D information
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("samples in X training set: "+str(len(X_train)))
print("sammples in test set: "+str(len(X_test)))
#print(X)
print("labels in test set:")
print(y_test)

from sklearn.neighbors import KNeighborsClassifier

def doKnn(X_train,y_train,X_test,y_test):
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict Output
    predictions = knn.predict(X_test)

    print("Evaluating performance: Confusion matrix")
    matrix = confusion_matrix(y_test, predictions)  # compare
    print(matrix)
    # output is the confusion matrix

    print(classification_report(y_test, predictions))

    print("precision: " + str(precision_score(y_test, predictions, average='weighted')))
    print("recall: " + str(recall_score(y_test, predictions, average='weighted')))

    print("f1 score " + str(f1_score(y_test, predictions, average='weighted')))


print("******** KNN på fuldt iris datasæt **************")
doKnn(X_train,y_train,X_test,y_test)


selector = SelectKBest(k=2)
X_train_reduced = selector.fit_transform(X_train, y_train)
print("reduced shape - selectKbest = 2")
print(X_train_reduced.shape)
print("feature scores")
print(selector.scores_)

features_selected = selector.get_support(indices=True)
print("indexes selected as best: ")
print(features_selected)
print("feature labels selected:")
for i in features_selected:
    print(iris.feature_names[i])



#do with the 2 worst features - kommenterer de 3 linjer ud
print("******** KNN på dårligste features **************")
X_train_reduced = np.hstack([X_train[:,0:1],X_train[:,1:2]])
X_test_reduced = np.hstack([X_test[:,0:1],X_test[:,1:2]])
doKnn(X_train_reduced,y_train,X_test_reduced,y_test)


print("******** KNN på bedste features **************")
X_train_reduced = np.hstack([X_train[:,2:3],X_train[:,3:4]])
X_test_reduced = np.hstack([X_test[:,2:3],X_test[:,3:4]])

doKnn(X_train_reduced,y_train,X_test_reduced,y_test)

# best 2 features, F1 = 1.0, Worst 2 features, F1 = 0.76
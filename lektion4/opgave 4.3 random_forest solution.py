
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
print("size of x_train = "+str(len(X_train)))
print("size of y_train = "+str(len(y_train)))
print("size of x_test = "+str(len(X_test)))
print("size of y_test = "+str(len(y_test)))

#clf = RandomForestClassifier(n_estimators=20)  # using 20 trees

accuracy = []
error = []
# training - with different number of trees - from 1 til 70
for i in range(1,50):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train,y_train)
    acc= clf.score(X_test,y_test)
    accuracy.append(acc)

plt.figure(figsize=(8,8))
plt.plot(accuracy,label='Accuracy')
plt.legend()
plt.title("RandomForest training - different number of trees")
plt.xlabel("Number of Trees used")
plt.show()
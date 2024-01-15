import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
test=pd.read_csv("Testing.csv") 
train=pd.read_csv("Training.csv")
data = pd.concat([train, test])
X, y=data.iloc[:,:-1], data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
sfs = SequentialFeatureSelector(rf_classifier, k_features=20, forward=True, scoring='accuracy', cv=5)
sfs.fit(X_train, y_train)
print("Selected feature indices:", sfs.k_feature_idx_)
X_train_selected = sfs.transform(X_train)
X_test_selected = sfs.transform(X_test)
rf_classifier.fit(X_train_selected, y_train)
y_pred = rf_classifier.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)
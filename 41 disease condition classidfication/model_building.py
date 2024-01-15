from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
import pandas as pd
train = pd.read_csv('Training.csv')
test = pd.read_csv('Testing.csv')
data = pd.concat([train, test])
X, y=data.iloc[:,:-1], data.iloc[:,-1]
## These are the top 20 features gave by the Sequential Feature Selector
column_indices = [0, 1, 6, 7, 11, 12, 14, 25, 31, 34, 35, 40, 41, 50, 56, 74, 81, 85, 97, 101]
selected_columns = data.columns[column_indices]
print("Selected Column Names:", list(selected_columns))
reduced_df = data.iloc[:, column_indices]
X=reduced_df
X_train, X_test, y_train, y_test = train_test_split(reduced_df, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:with features ", accuracy)
model_filename = "20featureswithaccuracy(0.93).pkl"  # Replace with your desired file name
with open(model_filename, 'wb') as file:
    pickle.dump(rf_classifier, file)

print(f"Model saved to {model_filename}")

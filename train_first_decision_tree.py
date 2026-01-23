from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
import pandas as pd

df = pd.read_csv("titanic.csv")

X = df.drop(columns=["Survived"]) #We drop what we want to predict
Y = df["Survived"]

X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True) 
X = X.drop(columns=["Name", "Ticket", "Cabin"])

X["Age"] = X["Age"].fillna(X["Age"].median()) #Handle the missing values in the csv


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

clf = DecisionTree(max_depth=10)
clf.fit(X_train.values, Y_train.values)
prediction = clf.predict(X_test.values)

def accuracy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)


acc = accuracy(prediction, Y_test)

print(acc)

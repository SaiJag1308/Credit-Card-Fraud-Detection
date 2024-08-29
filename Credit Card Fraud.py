# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
# Load the dataset from the csv file using pandas

data = pd.read_csv("creditcard.csv")
print(data.shape)
print(data.describe())

# Determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(fraud)))
print('Fraud Cases: {}'.format(len(valid)))
print('\n')

# Amount details of the fraudulent transaction
print('Amount details of the fraudulent transaction: \n{}'.format(fraud.Amount.describe()))
print('\n')

# Amount details of the valid transaction
print('Amount details of the valid transaction: \n{}'.format(valid.Amount.describe()))

# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(10,9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

# dividing the X and the Y from the dataset
X = data.drop(['Class'], axis = 1)
Y = data["Class"]
print(X.shape)
print(Y.shape)
# getting just the values for the sake of processing
xData = X.values
yData = Y.values

# Using Scikit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42)

# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
# predictions
yPred = rfc.predict(xTest)

# Evaluating the classifier
# printing every score of the classifier
# scoring in anything
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef

n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()
print("The model used is Random Forest classifier")

acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc))

prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec))

rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec))

f1 = f1_score(yTest, yPred)
print("The F1-Score is {}".format(f1))

MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is {}".format(MCC))

# Building the Decision tree Classifier (DECISION TREE)
from sklearn.tree import DecisionTreeClassifier
# dividing the X and the Y from the dataset
X = data.drop(['Class'], axis = 1)
Y = data["Class"]
print(X.shape)
print(Y.shape)
# getting just the values for the sake of processing
xData = X.values
yData = Y.values
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42)

# Creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
clf_gini.fit(xTrain, yTrain)

# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,max_depth=3, min_samples_leaf=5)

# Performing training
clf_entropy.fit(xTrain, yTrain)
print("The model used is Decision Tree classifier")
y_pred = clf_gini.predict(xTest)
print("Accuracy : ",accuracy_score(yTest, y_pred))
print("Precision Score : ",precision_score(yTest, y_pred))
print("Recall Score : ",recall_score(yTest, y_pred))
print("F1 Score : ",f1_score(yTest, y_pred))
print("Mathew's Correlation Coefficient : ",matthews_corrcoef(yTest, y_pred))
print("Report : ",classification_report(yTest, y_pred))
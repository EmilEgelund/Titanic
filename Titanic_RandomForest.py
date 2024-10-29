import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('titanic_800.csv', sep=',', header=0)

# Handle missing values
data["Age"].fillna(data["Age"].mean(), inplace=True)
data['Cabin'].fillna('Unknown', inplace=True)
data['Deck'] = data['Cabin'].apply(lambda x: x[0])
data.drop('Cabin', axis=1, inplace=True)

# Convert categorical features to numerical
data['Deck'] = data['Deck'].astype('category').cat.codes
data['Sex'] = data['Sex'].astype('category').cat.codes
data['Embarked'] = data['Embarked'].astype('category').cat.codes

# Drop unused features
data.drop(['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1, inplace=True)

# Separate features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=0)

# Train the RandomForestClassifier with 30 trees
clf = RandomForestClassifier(n_estimators=30, random_state=0)
clf.fit(xtrain, ytrain)

# Make predictions on the test set
predictions = clf.predict(xtest)

# Generate and print the confusion matrix and classification report
matrix = confusion_matrix(ytest, predictions)
print("Confusion Matrix:")
print(matrix)

target_names = ['Not Survived', 'Survived']
report = classification_report(ytest, predictions, target_names=target_names)
print("\nClassification Report:")
print(report)

accuracy = []
error = []
# training - with different number of trees - from 1 til 50
for i in range(1,50):
    clf = RandomForestClassifier(n_estimators=i, random_state=0)
    clf.fit(xtrain,ytrain)
    acc= clf.score(xtest,ytest)
    accuracy.append(acc)

plt.figure(figsize=(8,8))
plt.plot(accuracy,label='Accuracy')
plt.legend()
plt.title("RandomForest training - different number of trees")
plt.xlabel("Number of Trees used")
plt.show()
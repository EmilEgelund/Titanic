import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('titanic_800.csv', sep=',', header = 0)

#Replacing unknown data:
#Mean age is 29
data["Age"].fillna(29, inplace=True)

#Changing 'Cabin' to 'Deck', and dropping 'Cabin'
data['Cabin'].fillna('Unknown', inplace=True)
data['Deck'] = data['Cabin'].apply(lambda x: x[0])
data.drop('Cabin', axis=1, inplace=True)

# Converting categorial features to numerical
data['Deck'] = data['Deck'].astype('category').cat.codes
data['Sex'] = data['Sex'].astype('category').cat.codes
data['Embarked'] = data['Embarked'].astype('category').cat.codes

#Drop unused features
data.drop('PassengerId', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)


#Copy 'Survived' into yvalues and delete from data.
yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
yvalues["Survived"] = data["Survived"].copy()
data.drop('Survived', axis=1, inplace=True)

# 80/20 split between training and testing.
xtrain = data.head(640)
xtest = data.tail(160)

ytrain = yvalues.head(640)
ytest = yvalues.tail(160)

# Scaling the data
scaler = StandardScaler()
scaler .fit(xtrain)
xtrain = scaler .transform(xtrain)
xtest = scaler .transform(xtest)


# Experiment with different configurations
# High accuracy ones
mlp_configs = [
    {'hidden_layer_sizes': (12, 12, 12), 'activation': 'logistic', 'max_iter': 150, 'solver': 'lbfgs', 'learning_rate_init': 0.005},
    {'hidden_layer_sizes': (12, 12, 12), 'activation': 'logistic', 'max_iter': 150, 'solver': 'lbfgs', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (12, 12, 12), 'activation': 'logistic', 'max_iter': 150, 'solver': 'lbfgs', 'learning_rate_init': 0.0025},
]
# Lower accuracy ones
mlp_configs2 = [
    {'hidden_layer_sizes': (10, 10, 10), 'activation': 'relu', 'max_iter': 200, 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (8, 8, 8, 8), 'activation': 'tanh', 'max_iter': 300, 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (6, 6, 6, 6, 6), 'activation': 'relu', 'max_iter': 250, 'solver': 'adam', 'learning_rate_init': 0.0005},
    {'hidden_layer_sizes': (10, 10, 10, 10), 'activation': 'tanh', 'max_iter': 200, 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (5, 5, 5), 'activation': 'identity', 'max_iter': 200, 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (15, 15, 15), 'activation': 'relu', 'max_iter': 300, 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (20, 20, 20), 'activation': 'tanh', 'max_iter': 150, 'solver': 'lbfgs', 'learning_rate_init': 0.005},
    {'hidden_layer_sizes': (25, 25, 25), 'activation': 'logistic', 'max_iter': 250, 'solver': 'adam', 'learning_rate_init': 0.0005},
    {'hidden_layer_sizes': (30, 30, 30), 'activation': 'relu', 'max_iter': 200, 'solver': 'adam', 'learning_rate_init': 0.001},
]

# Iterate over configurations and evaluate each
for config in mlp_configs:
    mlp = MLPClassifier(**config, random_state=0)
    mlp.fit(xtrain, ytrain.values.ravel())
    predictions = mlp.predict(xtest)
    matrix = confusion_matrix(ytest, predictions)
    print(matrix)
    target_names = ['Not Survived', 'Survived']
    print(classification_report(ytest, predictions, target_names=target_names))
    tn, fp, fn, tp = matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Config: {config}, Accuracy: {accuracy * 100:.2f}%")


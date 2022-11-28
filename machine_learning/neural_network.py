from datetime import datetime
import pickle

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

df = pd.read_csv('../game/data/1669567704/72_dino_8.csv').dropna()

x = df.drop('action', axis=1)
y = df['action']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

sc=StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)

mlp_clf = MLPClassifier(hidden_layer_sizes=(7,1), max_iter = 50,activation = 'relu', solver = 'adam', shuffle = True)
mlp_clf.fit(trainX_scaled, trainY)

#https://python-course.eu/machine-learning/neural-networks-with-scikit.php

print("weights between input and first hidden layer:")
print(mlp_clf.coefs_[0])
print("\nweights between first hidden and second hidden layer:")
print(mlp_clf.coefs_[1])

print("Bias values for first hidden layer:")
print(mlp_clf.intercepts_[0])
print("\nBias values for second hidden layer:")
print(mlp_clf.intercepts_[1])


#
#y_pred = mlp_clf.predict(testX_scaled)
#
#print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

#fig = plot_confusion_matrix(mlp_clf, testX_scaled, testY, display_labels=mlp_clf.classes_)
#fig.figure_.suptitle("Confusion Matrix for Winequality Dataset")
#plt.show()
#
#plt.plot(mlp_clf.loss_curve_)
#plt.title("Loss Curve", fontsize=14)
#plt.xlabel('Iterations')
#plt.ylabel('Cost')
#plt.show()

#dt = datetime.now()
#ts = datetime.timestamp(dt)
#model_name = 'models/model_'+str(int(ts))+'.sav'
#pickle.dump(mlp_clf, open(model_name, 'wb'))
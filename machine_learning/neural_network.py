from datetime import datetime
import pickle
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.base import clone



def suffle(mlp):

    # TODO change this process
    #TODO change also mlp_clf.coefs_[0] & mlp_clf.coefs_[1]

    #https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#interpreting-coefficients-scale-matters
    #https://python-course.eu/machine-learning/neural-networks-with-scikit.php
    #print("weights between input and first hidden layer:")
    #print(mlp.coefs_[0])
    #print("\nweights between first hidden and second hidden layer:")
    #print(mlp.coefs_[1])
    #print("Bias values for first hidden layer:")
    #print(mlp.intercepts_[0])
    #print("\nBias values for second hidden layer:")
    #print(mlp.intercepts_[1])
    #y_pred = mlp_clf.predict(testX_scaled)
    #print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
    #fig = plot_confusion_matrix(mlp_clf, testX_scaled, testY, display_labels=mlp_clf.classes_)
    #fig.figure_.suptitle("Confusion Matrix for Winequality Dataset")
    #plt.show()
    #plt.plot(mlp_clf.loss_curve_)
    #plt.title("Loss Curve", fontsize=14)
    #plt.xlabel('Iterations')
    #plt.ylabel('Cost')
    #plt.show()

    for i in range(len(mlp.intercepts_)):
        
        index_0=np.random.randint(0,len(mlp.intercepts_[i]))
        index_1=np.random.randint(0,len(mlp.intercepts_[i]))
        #bias0_0 = mlp.intercepts_[i][index_0]
        #bias0_1 = mlp.intercepts_[i][index_1]

        mlp.intercepts_[i][index_0], mlp.intercepts_[i][index_1] = mlp.intercepts_[i][index_1], mlp.intercepts_[i][index_0]

    return mlp
    
    

def create_default_df():        

    data = {'distance_next': [920,920,920],
            'x_next': [920, 920, 920],
            'y_next': [329,329,329],
            'width_next':[68,68,68],
            'height_next':[71,71,71],
            'y_dino':[340,340,340],
            'game_speed':[20,20,20],
            'action':[0,1,2]
            }
    return pd.DataFrame(data)
    
def train(iteration, timestamp, dino_number, csv_file):

    folder = 'machine_learning/models/'+str(int(timestamp))
    os.mkdir(folder)

    df = None
    if csv_file:
        df = pd.read_csv(csv_file).dropna()
    else:
        df = create_default_df()

    x = df.drop('action', axis=1)
    y = df['action']

    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

    sc=StandardScaler()

    scaler = sc.fit(trainX)
    trainX_scaled = scaler.transform(trainX)
    testX_scaled = scaler.transform(testX)

    mlp_clf = MLPClassifier(hidden_layer_sizes=(7,1), max_iter = 50,activation = 'relu', solver = 'adam', shuffle = True)
    mlp_clf.fit(trainX_scaled, trainY)

    mlp_clf_=mlp_clf
    for i in range(dino_number):

        #first iteration does not vary
        if i != 0:
            mlp_clf_ = clone(mlp_clf)
            mlp_clf_.fit(trainX_scaled, trainY)
            mlp_clf_ = suffle(mlp_clf_)

        model_name = folder + '/model_'+str(i)+'.sav'
        pickle.dump(mlp_clf_, open(model_name, 'wb'))

    return folder
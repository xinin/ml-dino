import pickle
import os

import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

def create_default_df():        

    data = {
        'distance_next': np.random.randint(0, 1000, size=1000),
        'y_next': np.random.randint(0, 200, size=1000),
        'width_next': np.random.randint(50, 100, size=1000),
        'height_next': np.random.randint(50, 200, size=1000),
        'y_dino':np.random.randint(0, 200, size=1000),
        'game_speed': np.random.randint(20, 100, size=1000),
        'action':np.random.randint(0, 3, size=1000)
    }
    return pd.DataFrame(data)

def generate_first_generation(folder, dino_number):
    df = create_default_df()

    x = df.drop('action', axis=1)
    y = df['action']

    sc=StandardScaler()
    scaler = sc.fit(x)
    X_scaled = scaler.transform(x)


    # Crear una red neuronal para el ejemplo
    model = Sequential()
    model.add(Dense(10, input_shape=(6,), activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    #mlp_clf = MLPClassifier(hidden_layer_sizes=(7,6,5,4,3), max_iter = 1000, solver = 'lbfgs', activation='relu')
    #mlp_clf = MLPClassifier(hidden_layer_sizes=(7,3), max_iter = 1000, solver = 'lbfgs', activation='relu')
   
    for i in range(dino_number):
        print("Dino Model "+str(i))
        mlp_clf_ = clone_model(model)
        mlp_clf_.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        mlp_clf_.fit(X_scaled, y, epochs=1, batch_size=32)

        model_name = folder + '/model_'+str(i)+'.sav'
        mlp_clf_.save(model_name)
        #pickle.dump(mlp_clf_, open(model_name, 'wb'))

def mutate(child, iteration, dynamic_mutation, mutation_based_on_score, max_score):

    pass

def reproduce(parent_model, mother_model, iteration, dynamic_mutation, mutation_based_on_score, max_score):

    pass
        
def learn_from_parents(child, parent, mother):
    
    pass

def choose_two_with_bias(arr):
    #weights = [i+1 for i in range(len(arr))]
    #weights.reverse()

    weights = [int(a['score'].split("/")[-1].split("_")[0]) for a in arr]
    #print("weights",weights)
    choices = random.choices(arr, weights=weights, k=2)
    return choices[0],choices[1]



def generate_brains(iteration, timestamp, dino_child_number, best_dinos, dynamic_mutation, parents_in_generation, mutation_based_on_score, max_score, use_parent_knowledge):
    folder = 'models/'+str(int(timestamp))+'/'+str(iteration)
    os.mkdir(folder)

    if iteration == 0:
        generate_first_generation(folder, dino_child_number)
    else:

        pass

    return folder
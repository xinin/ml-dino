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

    heritage_percentage = np.random.randint(11)*0.2
    
    
    child = keras.models.clone_model(parent_model)
    for capa1, capa2 in zip(parent_model.layers, mother_model.layers):
        if isinstance(capa1, Dense):
            pesos1 = capa1.get_weights()
            pesos2 = capa2.get_weights()
            nuevos_pesos = []
            for matriz1, matriz2 in zip(pesos1, pesos2):
                forma = matriz1.shape
                mascara = np.random.choice([0, 1], size=forma)
                nuevos_pesos.append(matriz1 * mascara + matriz2 * (1 - mascara))
            capa1.set_weights(nuevos_pesos)
    
    return child
    
    #for i in range(min(len(parent_model.intercepts_),len(parent_model.intercepts_))):
    #    for j in range(min((len(parent_model.intercepts_[i])),len(mother_model.intercepts_[i]))):
    #        if np.random.random() < heritage_percentage:
    #            parent_model.intercepts_[i][j] = mother_model.intercepts_[i][j]

    #for i in range(len(parent_model.coefs_)):
    #    for j in range((len(parent_model.coefs_[i]))):
    #        if np.random.random() < heritage_percentage:
    #            mother_model.coefs_[i][j] = mother_model.coefs_[i][j]

    #if dynamic_mutation:
    #    if np.random.randint(0,100) <= max(100/iteration, 10):
    #        return mutate(parent_model, iteration, dynamic_mutation, mutation_based_on_score, max_score)
    #    else:
    #        return parent_model
    #else:
    #    if np.random.randint(0,100) <= 50:
    #        return mutate(parent_model, iteration, dynamic_mutation, mutation_based_on_score, max_score)
    #    else:
    #        return parent_model
        
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
        
        if parents_in_generation:
            for i,pd in enumerate(best_dinos):
                
                parent = tf.keras.models.load_model(pd['model'])
                #parent = pickle.load(open(pd['model'], 'rb'))
                
                model_name = folder + '/model_'+str(i)+'.sav'
                parent.save(model_name)
                #pickle.dump(parent, open(model_name, 'wb'))    
                
                
            for i in range(len(best_dinos),dino_child_number):

                parent,mother = choose_two_with_bias(best_dinos)
                
                parent_model = tf.keras.models.load_model(parent['model'])
                mother_model = tf.keras.models.load_model(mother['model'])
                #parent_model = pickle.load(open(parent['model'], 'rb'))
                #mother_model = pickle.load(open(mother['model'],'rb'))

                child = reproduce(parent_model, mother_model,iteration,dynamic_mutation, mutation_based_on_score, max_score)
                
                if use_parent_knowledge:
                    child = learn_from_parents(child, parent, mother)

                model_name = folder + '/model_'+str(i)+'.sav'
                child.save(model_name)
                #pickle.dump(child, open(model_name, 'wb'))
        else:
            #parent = len(best_dinos)-1
            for i in range(dino_child_number):
                
                parent,mother = choose_two_with_bias(best_dinos)
                parent_model = tf.keras.models.load_model(parent['model'])
                mother_model = tf.keras.models.load_model(mother['model'])
                #parent_model = pickle.load(open(parent['model'], 'rb'))
                #mother_model = pickle.load(open(mother['model'],'rb'))

                child = reproduce(parent_model, mother_model,iteration,dynamic_mutation, mutation_based_on_score, max_score)
                if use_parent_knowledge:
                    #child = learn_from_parents(child, best_dinos[parent], best_dinos[mother])
                    child = learn_from_parents(child, parent, mother)
                
                model_name = folder + '/model_'+str(i)+'.sav'
                child.save(model_name)
                #pickle.dump(child, open(model_name, 'wb'))

    return folder
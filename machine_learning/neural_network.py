import pickle
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone

def create_default_df():        

    data = {'distance_next': [920,920,100,920,920,100,920,920,100,920,920,100,920,920,100,920,920,100,920,920,100],
            'y_next': [329,329,329,329,329,329,329,329,329,329,329,329,329,329,329,329,329,329,329,329,329],
            'width_next':[68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68],
            'height_next':[71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71],
            'y_dino':[340,340,340,340,340,340,340,340,340,340,340,340,340,340,340,340,340,340,340,340,340],
            'game_speed':[20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20],
            'action':[0,1,0,0,1,0,0,1,2,0,1,0,0,1,2,0,1,0,0,1,2]
            }
    return pd.DataFrame(data)

def generate_first_generation(folder, dino_number):
    df = create_default_df()

    x = df.drop('action', axis=1)
    y = df['action']

    sc=StandardScaler()
    scaler = sc.fit(x)
    X_scaled = scaler.transform(x)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(7,3), max_iter = 500, solver = 'lbfgs')
   
    for i in range(dino_number):
        mlp_clf_ = clone(mlp_clf)
        mlp_clf_.fit(X_scaled, y)

        model_name = folder + '/model_'+str(i)+'.sav'
        pickle.dump(mlp_clf_, open(model_name, 'wb'))


def mutate(child, iteration, dynamic_mutation):

    # A higher mutation_rate leads to longer stagnation at the beginning, but leads to faster game progressing in the long run
    # A lower mutation_rate leads to faster initial progress, but to slower longterm progress.
    # Any mutation_rate higher then 0.09 leads to longterm stagnation.
    

    if dynamic_mutation:
         mutation_rate= max(0.5/iteration,0.05) 
         mutation_magnitude= max(1/iteration, 0.1)
         print("mutation_rate", mutation_rate)
         print("mutation_magnitude", mutation_magnitude)
    else:
        mutation_rate = 0.05
        mutation_magnitude = 0.1

    for i in range(len(child.intercepts_)):
        for j in range((len(child.intercepts_[i]))):
            if np.random.random() < mutation_rate:
                        child.intercepts_[i][j] = np.random.randn() * mutation_magnitude

    for i in range(len(child.coefs_)):
        for j in range((len(child.coefs_[i]))):
                if np.random.random() < mutation_rate:
                        child.coefs_[i][j] = np.random.randn() * mutation_magnitude

    return child

def reproduce(parent_model, mother_model, iteration, dynamic_mutation):

    heritage_percentage = np.random.randint(11)*0.2

    for i in range(len(parent_model.intercepts_)):
        for j in range((len(parent_model.intercepts_[i]))):
            if np.random.random() < heritage_percentage:
                parent_model.intercepts_[i][j] = mother_model.intercepts_[i][j]

    for i in range(len(parent_model.coefs_)):
        for j in range((len(parent_model.coefs_[i]))):
            if np.random.random() < heritage_percentage:
                mother_model.coefs_[i][j] = mother_model.coefs_[i][j]

    if dynamic_mutation:
        if np.random.randint(0,100) <= max(100/iteration, 10):
            return mutate(parent_model, iteration, dynamic_mutation)
        else:
            return parent_model
    else:
        if np.random.randint(0,100) <= 30:
            return mutate(parent_model, iteration, dynamic_mutation)
        else:
            return parent_model

def generate_brains(iteration, timestamp, dino_child_number, best_dinos, dynamic_mutation):
    folder = 'models/'+str(int(timestamp))
    os.mkdir(folder)

    if iteration == 0:
        generate_first_generation(folder, dino_child_number)
    else:
        #parent = len(best_dinos)-1
        for i in range(dino_child_number):
            parent = np.random.randint(0,len(best_dinos))
            parent_model = pickle.load(open(best_dinos[parent]['model'], 'rb'))
            mother = np.random.randint(0,len(best_dinos))
            mother_model = pickle.load(open(best_dinos[mother]['model'],'rb'))

            #df = pd.concat([pd.read_csv(best_dinos[parent]['score']).dropna(),pd.read_csv(best_dinos[mother]['score']).dropna()])                  
            #x = df.drop('action', axis=1)
            #y = df['action']
            #sc=StandardScaler()
            #scaler = sc.fit(x)
            #X_scaled = scaler.transform(x)
            #parent_model.fit(X_scaled, y)

            child = reproduce(parent_model, mother_model,iteration,dynamic_mutation)

            model_name = folder + '/model_'+str(i)+'.sav'
            pickle.dump(child, open(model_name, 'wb'))
    return folder
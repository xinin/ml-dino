from game.game import Game
from datetime import datetime
#from machine_learning.neural_network import generate_brains
from machine_learning.neural_network_tensorflow import generate_brains
from game.data_colector import DataCollector
import os
import shutil
import tensorflow as tf
import csv
import pandas as pd

DINO_NUMBER = 200
REPRODUCTION_LEVEL = 10
ITERATIONS = 600
DYNAMIC_MUTATION = True
MUTATION_BASED_ON_SCORE = True
PARENTS_IN_GENERATION = True
USE_PARENT_KNOWLEDGE = True
BEST_ALL_TIME_IN_GENERATION = True
LEARN_FROM_ALL_THE_BEST = True
REAL_DEATH = True
REAL_DEATH_ITERATIONS = 5
MAX_TIME = 40000
REWARD_FUNC_TYPE = 1 #1 adds up if they survive for the time. 2, subtract if they do actions that do not make sense.
IMPROVISED_RATIO = 0.3 # value between 0 and 1

DATA_FOLDER = 'data/'
MODELS_FOLDER = 'models/'
MAX_SCORE_FOLDER = 'max_score/'
DATA_TRAINING = 'training_data/'

#shutil.rmtree(DATA_FOLDER, ignore_errors=True)
#os.mkdir(DATA_FOLDER)

#shutil.rmtree(MODELS_FOLDER, ignore_errors=True)
#os.mkdir(MODELS_FOLDER)

shutil.rmtree(MAX_SCORE_FOLDER, ignore_errors=True)
os.mkdir(MAX_SCORE_FOLDER)
max_score = 0
with open(MAX_SCORE_FOLDER+'score', 'w') as f:
    f.write(str(max_score))

best_dinos_all_time = []
best_dinos = []

dt = datetime.now()
ts = datetime.timestamp(dt)
os.mkdir(DATA_FOLDER+str(int(ts)))
os.mkdir(DATA_TRAINING+str(int(ts)))
os.mkdir(MODELS_FOLDER+str(int(ts)))

for i in range(ITERATIONS):
    print("Iteration: "+str(i))
    print("Best")
    for item in best_dinos_all_time:
        score_number = item['score_number']
        training_data = item['training_data']
        print(f"score_number: {score_number}, training_data: {training_data}")
    
    with open(MAX_SCORE_FOLDER+'score', 'r') as f:
        max_score=f.read()
    
    ml_model_version = generate_brains(i,ts,DINO_NUMBER, best_dinos_all_time, DYNAMIC_MUTATION, PARENTS_IN_GENERATION, MUTATION_BASED_ON_SCORE, max_score, USE_PARENT_KNOWLEDGE, LEARN_FROM_ALL_THE_BEST)
    Game.init(i, ts, DINO_NUMBER, int(max_score), ml_model_version, REAL_DEATH, MAX_TIME, REAL_DEATH_ITERATIONS, REWARD_FUNC_TYPE, IMPROVISED_RATIO)
    
    scores = os.listdir(DATA_FOLDER+str(int(ts))+'/'+ str(i))
    scores = sorted(scores, key=lambda x: int(x.split('_')[0]), reverse=False)
    
    best_scores = scores[-REPRODUCTION_LEVEL:]
    best_scores.reverse()

    best_dinos = []
    for x in best_scores:
        
        score = {
                 'score_number': int(x.split('_')[0]),
                 'score': DATA_FOLDER + str(int(ts)) + '/' + str(i) + '/' + x,
                 'training_data': DATA_TRAINING + str(int(ts)) + '/' + str(i) + '/' + x,
                 'model': MODELS_FOLDER + str(int(ts)) + '/' + str(i) + '/model_' + x.split('_')[2].split('.')[0] + '.sav'
                }

        DataCollector.delete_failures(score['score'], score['training_data'])
    
        best_dinos.append(score)

    if BEST_ALL_TIME_IN_GENERATION :
        best_dinos_all_time += best_dinos
        # ordenar los objetos en base al atributo score_number
        best_dinos_all_time = sorted(best_dinos_all_time, key=lambda obj: obj['score_number'], reverse=True)[:REPRODUCTION_LEVEL]
    else:
        best_dinos_all_time = best_dinos
        
    if LEARN_FROM_ALL_THE_BEST:
        dataset = []

        for item in best_dinos_all_time:
            training_data_path = item['training_data']
            with open(training_data_path, 'r',newline='') as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)
                for row in csv_reader:
                    dataset.append(row)
                    
        headers = ['distance_next', 'y_next', 'width_next', 'height_next', 'y_dino', 'game_speed', 'action', 'score', 'last_failed']
        df = pd.DataFrame(dataset, columns=headers)
        df.to_csv(DATA_TRAINING+'best_dinos_dataset.csv', index=False)
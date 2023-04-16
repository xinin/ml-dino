from game.game import Game
from datetime import datetime
from machine_learning.neural_network import generate_brains
from game.data_colector import DataCollector
import os
import shutil

DINO_NUMBER = 600
REPRODUCTION_LEVEL = 10
ITERATIONS = 21
DYNAMIC_MUTATION = True
MUTATION_BASED_ON_SCORE = True
PARENTS_IN_GENERATION = True
USE_PARENT_KNOWLEDGE = True
REAL_DEATH = True
REAL_DEATH_ITERATIONS = 5
MAX_TIME = 40000

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

best_dinos = []

dt = datetime.now()
ts = datetime.timestamp(dt)
os.mkdir(DATA_FOLDER+str(int(ts)))
os.mkdir(DATA_TRAINING+str(int(ts)))
os.mkdir(MODELS_FOLDER+str(int(ts)))

for i in range(ITERATIONS):
    print("Iteration: "+str(i))
    with open(MAX_SCORE_FOLDER+'score', 'r') as f:
        max_score=f.read()
    
    ml_model_version = generate_brains(i,ts,DINO_NUMBER, best_dinos, DYNAMIC_MUTATION, PARENTS_IN_GENERATION, MUTATION_BASED_ON_SCORE, max_score, USE_PARENT_KNOWLEDGE)
    Game.init(i, ts, DINO_NUMBER, int(max_score), ml_model_version, REAL_DEATH, MAX_TIME, REAL_DEATH_ITERATIONS)
    
    scores = os.listdir(DATA_FOLDER+str(int(ts))+'/'+ str(i))
    scores = sorted(scores, key=lambda x: int(x.split('_')[0]), reverse=False)
    
    best_scores = scores[-REPRODUCTION_LEVEL:]
    best_scores.reverse()

    print(best_scores)

    best_dinos = []
    for x in best_scores:
        
        score = {
                 'score': DATA_FOLDER + str(int(ts)) + '/' + str(i) + '/' + x,
                 'training_data': DATA_TRAINING + str(int(ts)) + '/' + str(i) + '/' + x,
                 'model': MODELS_FOLDER + str(int(ts)) + '/' + str(i) + '/model_' + x.split('_')[2].split('.')[0] + '.sav'
                }

        DataCollector.delete_failures(score['score'], score['training_data'])
    
        best_dinos.append(score)


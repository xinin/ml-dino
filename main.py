from game.game import Game
from datetime import datetime
from machine_learning.neural_network import generate_brains
import os
import shutil

DINO_NUMBER = 400
REPRODUCTION_LEVEL = 20
ITERATIONS = 21
DYNAMIC_MUTATION = True

DATA_FOLDER = 'data/'
MODELS_FOLDER = 'models/'
MAX_SCORE_FOLDER = 'max_score/'

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
os.mkdir(MODELS_FOLDER+str(int(ts)))

for i in range(ITERATIONS):
    print("Iteration: "+str(i))
    ml_model_version = generate_brains(i,ts,DINO_NUMBER, best_dinos, DYNAMIC_MUTATION)
    with open(MAX_SCORE_FOLDER+'score', 'r') as f:
        max_score=f.read()
    Game.init(i, ts, DINO_NUMBER, int(max_score), ml_model_version)
    
    scores = os.listdir(DATA_FOLDER+str(int(ts))+'/'+ str(i))
    scores = sorted(scores, key=lambda x: int(x.split('_')[0]), reverse=False)
    
    best_dinos = []
    for x in scores[-REPRODUCTION_LEVEL:]:
        score = {'score': DATA_FOLDER + str(int(ts)) + '/' + str(i) + '/' + x,
                'model': MODELS_FOLDER + str(int(ts)) + '/' + str(i) + '/model_' + x.split('_')[2].split('.')[0] + '.sav'}
        best_dinos.append(score)
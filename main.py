from game.game import Game
from datetime import datetime
from machine_learning.neural_network import generate_brains
import os
import shutil

DINO_NUMBER = 300
REPRODUCTION_LEVEL = 10
ITERATIONS = 300

DATA_FOLDER = 'data/'
MODELS_FOLDER = 'models/'
MAX_SCORE_FOLDER = 'max_score/'

shutil.rmtree(DATA_FOLDER, ignore_errors=True)
os.mkdir(DATA_FOLDER)

shutil.rmtree(MODELS_FOLDER, ignore_errors=True)
os.mkdir(MODELS_FOLDER)

shutil.rmtree(MAX_SCORE_FOLDER, ignore_errors=True)
os.mkdir(MAX_SCORE_FOLDER)
max_score = 0
with open(MAX_SCORE_FOLDER+'score', 'w') as f:
    f.write(str(max_score))

best_dinos = []

for i in range(ITERATIONS):
    print("Iteration: "+str(i))
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    ml_model_version = generate_brains(i,ts,DINO_NUMBER, best_dinos)
    with open(MAX_SCORE_FOLDER+'score', 'r') as f:
        max_score=f.read()
    Game.init(i, ts, DINO_NUMBER, int(max_score), ml_model_version)
    
    scores = os.listdir(DATA_FOLDER+str(int(ts)))
    scores = sorted(scores, key=lambda x: int(x.split('_')[0]), reverse=False)
    best_dinos = list(map(lambda x: {'score':DATA_FOLDER+str(int(ts))+'/'+x, 'model':MODELS_FOLDER+str(int(ts))+'/model_'+x.split('_')[2].split('.')[0]+'.sav'},scores[-REPRODUCTION_LEVEL:]))
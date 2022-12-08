from game.game import Game
from datetime import datetime
from machine_learning.neural_network import train
import os
import shutil

DINO_NUMBER = 100
REPRODUCTION_LEVEL = 40
ITERATIONS = 100

csv_files = []

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

for i in range(ITERATIONS):
    print("Iteration: "+str(i))
    dt = datetime.now()
    ts = datetime.timestamp(dt)
#    print('csv_file',csv_files)
    ml_model_version = train(i,ts,DINO_NUMBER, csv_files)
    with open(MAX_SCORE_FOLDER+'score', 'r') as f:
        max_score=f.read()
    Game.init(i, ts, DINO_NUMBER, int(max_score), ml_model_version)
    
    #Get dinos behaviour with more score points
    #scores = os.listdir(DATA_FOLDER+str(int(ts)))
    #scores.sort()
    #csv_files = list(map(lambda x: DATA_FOLDER+str(int(ts))+'/'+x,scores[-REPRODUCTION_LEVEL:]))
    
    scores = os.listdir(DATA_FOLDER+str(int(ts)))
    scores.sort()
    csv_files = list(map(lambda x: DATA_FOLDER+str(int(ts))+'/'+x,scores[-REPRODUCTION_LEVEL:]))


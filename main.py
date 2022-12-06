from game.game import Game
from datetime import datetime
from machine_learning.neural_network import train
import os

DINO_NUMBER = 100
REPRODUCTION_LEVEL = 20
ITERATIONS = 100

csv_files = []

for i in range(ITERATIONS):
    print("Iteration: "+str(i))
    dt = datetime.now()
    ts = datetime.timestamp(dt)
#    print('csv_file',csv_files)
    ml_model_version = train(i,ts,DINO_NUMBER, csv_files)
    Game.init(i, ts, DINO_NUMBER, ml_model_version)
    #Get dinos behaviour with more score points
    scores = os.listdir('data/'+str(int(ts)))
    scores.sort()
    csv_files = list(map(lambda x: 'data/'+str(int(ts))+'/'+x,scores[-REPRODUCTION_LEVEL:]))
    
from game.game import Game
from datetime import datetime
from machine_learning.neural_network import train
import os

DINO_NUMBER = 10

ml_model = 'machine_learning/models/model_default.sav'
#csv_file = 'data/default/43_dino_1.csv'
csv_file = None

for i in range(10):
    print("Iteration: "+str(i))
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    print('csv_file',csv_file)
    ml_model_version = train(i,ts,DINO_NUMBER, csv_file)
    Game.init(i, ts, DINO_NUMBER, ml_model_version)
    #Get dino's behaviour with more score points
    scores = os.listdir('data/'+str(int(ts)))
    print(scores)
    scores.sort()
    print(scores)
    print(scores[-1])
    csv_file = 'data/'+str(int(ts))+'/'+scores[-1]
    


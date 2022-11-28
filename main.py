from game.game import Game
from datetime import datetime

DINO_NUMBER = 10

for i in range(2):

    dt = datetime.now()
    ts = datetime.timestamp(dt)

    ml_model = 'machine_learning/models/model_1669569137.sav'

    Game.init(i, ts, DINO_NUMBER, ml_model)
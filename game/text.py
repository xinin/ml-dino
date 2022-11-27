import pickle
import pandas as pd

loaded_model = pickle.load(open('../machine_learning/models/model_1669569137.sav', 'rb'))

input = [[340,580,332,97,68,272,20]]

pred = loaded_model.predict(input)
print(pred)
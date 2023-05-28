import pygame
import os
import pickle
import random
from game.constants import DEBUG
from sklearn.preprocessing import StandardScaler
import numpy as np
from io import BytesIO

import tensorflow as tf

class Dino:

    RUNNING = [
        pygame.image.load(os.path.join("game/assets/Dino", "DinoRun1.png")),
        pygame.image.load(os.path.join("game/assets/Dino", "DinoRun2.png")),
    ]
    JUMPING = [
        pygame.image.load(os.path.join("game/assets/Dino", "DinoJump.png")),
        pygame.image.load(os.path.join("game/assets/Dino", "DinoJump.png"))
    ]
    DUCKING = [
        pygame.image.load(os.path.join("game/assets/Dino", "DinoDuck1.png")),
        pygame.image.load(os.path.join("game/assets/Dino", "DinoDuck2.png")),
    ]

    def __init__(self, position_x, position_y, child_number, ml_model, IMPROVISED_RATIO):
        self.action = Dino.RUNNING
        self.initial_y = position_y
        self.initial_x = position_x
        self.rect = self.action[0].get_rect(bottomleft=(self.initial_x, self.initial_y))
        self.speed_y = 0
        self.steps = 0
        self.fails = 0
        self.useless_actions = 0
        self.usefull_actions = 0
        self.last_failed = False
        self.death = False
        self.child_number = child_number
        self.IMPROVISED_RATIO = IMPROVISED_RATIO
        #self.ml_model = pickle.load(open(ml_model, 'rb'))
        
        model = tf.keras.models.load_model(ml_model)
         
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Crear un objeto BytesIO y cargar el modelo TensorFlow Lite
        model_stream = BytesIO(tflite_model)

        # Crear un intérprete TensorFlow Lite desde los datos en memoria
        interpreter = tf.lite.Interpreter(model_content=model_stream.getvalue())
        
        self.ml_model = interpreter
        
        self.ml_model.allocate_tensors()
        
        input_details = self.ml_model.get_input_details()
        output_details = self.ml_model.get_output_details()
        
        self.input_details = input_details[0]['index']
        self.output_details = output_details[0]['index']

    def jump(self):
        if self.action != Dino.JUMPING:
            self.action = Dino.JUMPING
            self.speed_y = 7
            self.rect = self.action[0].get_rect(bottomleft=(self.initial_x, self.initial_y))

    def duck(self):
        if self.action != Dino.JUMPING:
            self.action = Dino.DUCKING
            self.rect = self.action[0].get_rect(bottomleft=(self.initial_x, self.initial_y))

    def running(self):
        if self.action != Dino.JUMPING:
            self.action = Dino.RUNNING
            self.rect = self.action[0].get_rect(bottomleft=(self.initial_x, self.initial_y))

    def draw(self, SCREEN):
                
        if self.action == Dino.JUMPING:            
            if self.speed_y > 0:
                self.rect.y -= (self.speed_y**2 + 7)
            else:
                self.rect.y += (self.speed_y**2)
            self.speed_y = self.speed_y - 1

            if abs(self.speed_y) == 7:
                self.action = Dino.RUNNING
                self.rect.y = self.initial_y
                self.rect = self.action[0].get_rect(bottomleft=(self.initial_x, self.initial_y))
                self.speed_y = 0
        
        self.steps +=1
        SCREEN.blit(self.action[self.steps % 2], self.rect)
        if DEBUG:
            pygame.draw.rect(SCREEN, (0,0,0), pygame.Rect(self.rect.x, self.rect.y, self.rect.width, self.rect.height),  2, 3)
        self.last_failed = False


    def die(self):
        self.death = True

    def fail(self):
        self.fails += 1
        self.last_failed = True

    def do_useless_action(self):
        self.useless_actions += 1
    
    def do_usefull_action(self):
        self.usefull_actions +=1

    def get_score(self):
        return self.steps - (self.fails *100) - (self.useless_actions * 5) #+ (self.usefull_actions * 5)
    
    
    def predict(self, data):
        # Obtener información de entrada y salida del modelo
        #input_details = self.ml_model.get_input_details()
        #output_details = self.ml_model.get_output_details()

        # Preparar los datos de entrada para la predicción
        #data = np.array(data).astype(np.float32)
        
        #print("Forma de entrada esperada:", input_details[0]['shape'])
        #print("Forma de entrada dada:", data.shape)

        #input_data = data.reshape(input_details[0]['shape'])
        #print(input_data)
        #print("Forma de entrada dada:", input_data.shape)


        # Asignar los datos de entrada al tensor de entrada del modelo
        #self.ml_model.set_tensor(input_details[0]['index'], np.array(data).astype(np.float32))
        self.ml_model.set_tensor(self.input_details, np.array(data).astype(np.float32))

        # Realizar la predicción
        self.ml_model.invoke()

        # Obtener los resultados de la predicción
        #output_data = self.ml_model.get_tensor(output_details[0]['index'])
        output_data = self.ml_model.get_tensor(self.output_details)
        #print(output_data)
        #predicted_label = np.argmax(output_data)  # Obtener la etiqueta predicha
        #print("Etiqueta predicha:", predicted_label)
        return output_data


    def think(self, obstacles, game_speed, iteration):
        data = []
        obs = sorted(obstacles, key=lambda x: x.rect.x)    
        for obstacle in obs:
           if obstacle.rect.x >= self.rect.x:
                data.append([
                    abs(self.rect.x - obstacle.rect.x),
                    obstacle.rect.y,
                    obstacle.rect.width,
                    obstacle.rect.height,
                    self.rect.y,
                    game_speed,
                ])
                break

        if len(data) == 1:
            
            
            #sc=StandardScaler()
            #scaler = sc.fit(data)
            #X_scaled = scaler.transform(data)
            
            #print("data",data)
            #print("X_scaled",X_scaled)
            
                      
            if (self.IMPROVISED_RATIO > 0):
                improvised_chance = random.random()
                if iteration > 0 and improvised_chance < self.IMPROVISED_RATIO/iteration:

                    #forzamos que salte
                    #pred = np.argmax(self.ml_model.predict_proba(data)[0])
                    pred = np.argmax(self.predict(data)[0])
                    if pred == 1:
                        return 2
                    else:
                        return pred
                    #cogemos el segundo mas probable 
                    #pred = self.ml_model.predict_proba(data)[0]
                    #max = np.argmax(pred)
                    #pred[max] = -np.inf
                    #return np.argmin(np.argmax(pred))
                else:
                    #return np.argmax(self.ml_model.predict_proba(data)[0])
                    return np.argmax(self.predict(data)[0])

            else:
                #return np.argmax(self.ml_model.predict_proba(data)[0])
                return np.argmax(self.predict(data)[0])
        else:
            return np.argmax(self.predict([[self.rect.x,0,0,0,self.rect.y, game_speed]])[0])
            #return np.argmax(self.ml_model.predict_proba([[self.rect.x,0,0,0,self.rect.y, game_speed]])[0])
            #return self.ml_model.predict([[self.rect.x,0,0,0,self.rect.y, game_speed]])[0]

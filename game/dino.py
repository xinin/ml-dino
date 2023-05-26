import pygame
import os
import pickle
import random
from game.constants import DEBUG
from sklearn.preprocessing import StandardScaler
import numpy as np

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
        self.ml_model = tf.keras.models.load_model(ml_model)

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

    def think(self, obstacles, game_speed):
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

            
            if (self.IMPROVISED_RATIO > 0):
                improvised_chance = random.random()
                if improvised_chance < self.IMPROVISED_RATIO:

                    #forzamos que salte
                    #pred = np.argmax(self.ml_model.predict_proba(data)[0])
                    pred = np.argmax(self.ml_model.predict(data)[0])
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
                    return np.argmax(self.ml_model.predict(data)[0])

            else:
                #return np.argmax(self.ml_model.predict_proba(data)[0])
                return np.argmax(self.ml_model.predict(data)[0])
        else:
            return np.argmax(self.ml_model.predict([[self.rect.x,0,0,0,self.rect.y, game_speed]])[0])
            #return np.argmax(self.ml_model.predict_proba([[self.rect.x,0,0,0,self.rect.y, game_speed]])[0])
            #return self.ml_model.predict([[self.rect.x,0,0,0,self.rect.y, game_speed]])[0]

import pygame, random, os
from pygame.locals import *

from game.constants import SCREEN_HEIGHT, SCREEN_WIDTH, WHITE, BLACK, MAX_SCORE_FOLDER, HIGH_BIRD_HEIGHT, LOW_BIRD_HEIGHT

from game.dino import Dino
from game.obstacle import SmallCactus, LargeCactus, Bird
from game.data_colector import DataCollector
import datetime


import numpy as np

class Game:

    def init(iteration, timestamp, dino_number, max_score, ml_model, real_death, max_time, max_iterations_death, REWARD_FUNC_TYPE, IMPROVISED_RATIO):
        #create data folder
        folder = 'data/'+str(int(timestamp))+'/'+str(iteration)
        os.mkdir(folder)

        start_time = datetime.datetime.now()

        pygame.init()
        pygame.display.set_caption('Diego Prieto Dino ML')

        font = pygame.font.Font('freesansbold.ttf', 20)

        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0 , 32)

        game_speed=20
        max_dinos = dino_number
        obstacles=[]
        dinos = []
        steps = 0
        steps_next_obstacle = 0
        dead_dinos = 0
        running_dinos = 0
        jumping_dinos = 0
        ducking_dinos = 0
        obstacle_count = 0
        INTERSECTION = False

        for i in range(max_dinos):
            dinos.append(Dino(SCREEN_WIDTH*0.20, SCREEN_HEIGHT*0.5, i,ml_model+'/model_'+str(i)+'.sav', IMPROVISED_RATIO) ) 

        #Run the game loop
        now = datetime.datetime.now()
        while dead_dinos < max_dinos and (not real_death or (now - start_time).total_seconds() < max_time):
            clock.tick(game_speed)
            INTERSECTION = False
            screen.fill(WHITE)
            array_text = [
                'Iteration: '+str(iteration),
                'Max Score: '+str(max_score),
                'Current Score: '+str(steps),
                'Dinos Alive: '+str(max_dinos-dead_dinos),
                'Game Speed: '+str(game_speed),
                'Dinos Running: '+str(running_dinos),
                'Dinos Jumping: '+str(jumping_dinos),
                'Dinos Ducking: '+str(ducking_dinos),
            ]
            for i,t in enumerate(array_text):
                text = font.render(t, True, BLACK, WHITE)
                textRect = text.get_rect()
                textRect.y += 35*i
                screen.blit(text, textRect)

            pygame.draw.line(screen, BLACK, (0, SCREEN_HEIGHT*0.5), (SCREEN_WIDTH, SCREEN_HEIGHT*0.5), 1)
            pygame.event.get()

            running_dinos = 0
            jumping_dinos = 0
            ducking_dinos = 0
         
            for obstacle in obstacles:
                if obstacle.rect.x<0:
                    obstacles.remove(obstacle)
                break

            for index, dino in enumerate(dinos):
                if not dino.death:
                    for obstacle in obstacles:
                        if dino.rect.colliderect(obstacle.rect):
                            if not real_death and iteration < max_iterations_death:
                                dino.fail()
                            else:
                                dino.die()
                                if max_score < dino.get_score():
                                    with open(MAX_SCORE_FOLDER+'score', 'w') as f:
                                        f.write(str(dino.get_score()))
                                dead_dinos += 1
                                DataCollector.rename_file(folder, index, dino.get_score())
                        if REWARD_FUNC_TYPE == 2:
                            if dino.rect.x + dino.rect.width > obstacle.rect.x and dino.rect.x < obstacle.rect.x+ obstacle.rect.width:
                               INTERSECTION = True

                if not dino.death:
                    action = dino.think(obstacles, game_speed)
                    if action == 0:
                        dino.running()
                        running_dinos +=1
                        #if REWARD_FUNC_TYPE:
                        #    dino.do_usefull_action()
                    elif action == 1:
                        dino.jump()
                        jumping_dinos +=1
                        if REWARD_FUNC_TYPE == 2 and INTERSECTION:
                            dino.do_useless_action()
                        else:
                            dino.do_usefull_action()
                    elif action == 2:
                        dino.duck()
                        ducking_dinos +=1
                        if REWARD_FUNC_TYPE == 2 and INTERSECTION:
                            dino.do_useless_action()
                        else:
                            dino.do_usefull_action()

                    DataCollector.write_data(folder+'/dino_'+str(index)+'.csv', dino, obstacles, game_speed, action, dino.get_score())
                    dino.draw(screen)

            if steps_next_obstacle == 0:
                steps_next_obstacle = np.random.randint(40,50)
                if len(obstacles) <= 2:
                    obstacle_count += 1
                    if obstacle_count%5 == 0:
                        #h = LOW_BIRD_HEIGHT if random.randint(0,3)<1 else HIGH_BIRD_HEIGHT
                        obstacles.append(Bird(SCREEN_WIDTH, HIGH_BIRD_HEIGHT))
                    elif random.randint(0, 10) >= 6:
                        obstacles.append(LargeCactus(SCREEN_WIDTH, SCREEN_HEIGHT*0.5))
                    elif random.randint(0, 10)>6:
                        #h = LOW_BIRD_HEIGHT if random.randint(0,3)<1 else HIGH_BIRD_HEIGHT
                        obstacles.append(Bird(SCREEN_WIDTH, LOW_BIRD_HEIGHT)) 
                    else:
                        obstacles.append(SmallCactus(SCREEN_WIDTH, SCREEN_HEIGHT*0.5))
            
            for obstacle in obstacles:
                obstacle.update(game_speed)
                obstacle.draw(screen)

            pygame.display.update()
            steps = steps + 1
            steps_next_obstacle -=1
            #aumentar la velocidad segun los puntos
            if steps % 100 == 0:
                game_speed = game_speed +0.5

            now = datetime.datetime.now()


        print("Tiempo acabado", (now - start_time).total_seconds())
        for index, dino in enumerate(dinos):
            if not dino.death:
                dino.die()
                if max_score < dino.get_score():
                    with open(MAX_SCORE_FOLDER+'score', 'w') as f:
                        f.write(str(dino.get_score()))
                    max_score = dino.get_score()
                dead_dinos += 1
                DataCollector.rename_file(folder, index, dino.get_score())

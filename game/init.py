import pygame, random, os
from pygame.locals import *
from datetime import datetime

from constants import SCREEN_HEIGHT, SCREEN_WIDTH, WHITE, BLACK

from dino import Dino
from obstacle import SmallCactus, LargeCactus, Bird
from data_colector import DataCollector

def init():

    #create data folder
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    folder = 'data/'+str(int(ts))
    os.mkdir(folder)

    pygame.init()
    pygame.display.set_caption('Diego Prieto Dino ML')

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0 , 32)

    game_speed=20
    obstacles=[]
    max_dinos = 10
    dinos = []
    steps = 0

    for i in range(max_dinos):
        dinos.append(Dino(SCREEN_WIDTH*0.20, SCREEN_HEIGHT*0.5))  

    dead_dinos = 0

    #Run the game loop
    while dead_dinos < max_dinos:
        screen.fill(WHITE)
        pygame.draw.line(screen, BLACK, (0, SCREEN_HEIGHT*0.5), (SCREEN_WIDTH, SCREEN_HEIGHT*0.5), 1)
        pygame.event.get()

        for obstacle in obstacles:
            if obstacle.rect.x<0:
                obstacles.remove(obstacle)

        for index, dino in enumerate(dinos):

            if not dino.death:
                for obstacle in obstacles:
                    if dino.rect.colliderect(obstacle.rect):
                        dino.die()
                        dead_dinos += 1
                        os.rename(folder+'/dino_'+str(index)+'.csv', folder+'/'+str(dino.steps)+'_dino_'+str(index)+'.csv')

            if not dino.death:
                action = random.randint(0, 2)
                if action == 0:
                    dino.running()
                elif action == 1:
                    dino.jump()
                elif action == 2:
                    dino.duck()

                DataCollector.write_data(folder+'/dino_'+str(index)+'.csv', dino, obstacles, game_speed, action)
                dino.draw(screen)

        if steps % 20 == 0:
            if len(obstacles) <= 2:
                if random.randint(0, 2):
                    obstacles.append(SmallCactus(SCREEN_WIDTH, SCREEN_HEIGHT*0.5))
                elif random.randint(0, 2):
                    obstacles.append(LargeCactus(SCREEN_WIDTH, SCREEN_HEIGHT*0.5))
                else:
                    h = SCREEN_HEIGHT*0.50 if random.randint(0,1) else SCREEN_HEIGHT*0.40
                    obstacles.append(Bird(SCREEN_WIDTH, h))
        
        for obstacle in obstacles:
            obstacle.update(game_speed)
            obstacle.draw(screen)

        pygame.display.update()
        clock.tick(game_speed)
        steps = steps + 1
        #aumentar la velocidad segun los puntos
        #game_speed = game_speed +0.05

init()
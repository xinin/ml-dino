import pygame, sys, random, os
from pygame.locals import *
from datetime import datetime

from constants import SCREEN_HEIGHT, SCREEN_WIDTH, WHITE, BLACK

from dino import Dino
from obstacle import SmallCactus, LargeCactus, Bird
from data_colector import DataCollector

def init():
    pygame.init()
    pygame.display.set_caption('Diego Prieto Dino ML')

    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0 , 32)

    #Run the game loop
    #steps=0
    game_speed=20
    obstacles=[]
    death = False
    max_dinos = 1
    dinos = []

    for i in range(max_dinos):
        dinos.append(Dino(SCREEN_WIDTH*0.20, SCREEN_HEIGHT*0.5))  

    #create data folder
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    folder = 'data/'+str(int(ts))
    os.mkdir(folder)

    while death == False:
        screen.fill(WHITE)
        pygame.draw.line(screen, BLACK, (0, SCREEN_HEIGHT*0.5), (SCREEN_WIDTH, SCREEN_HEIGHT*0.5), 1)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                userInput=pygame.key.get_pressed()
                if (userInput[pygame.K_UP] or userInput[pygame.K_SPACE]):
                    dinos[0].jump()
                elif (userInput[pygame.K_DOWN]):
                    dinos[0].duck()  
                else:
                    dinos[0].running()    
            else:
                dinos[0].running()      

        for obstacle in obstacles:
            if obstacle.rect.x<0:
                obstacles.remove(obstacle)

            if dinos[0].rect.colliderect(obstacle.rect):
                #pygame.time.delay(2000)
                death = True

        dinos[0].draw(screen)

        if dinos[0].steps % 20 == 0:
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

        DataCollector.write_data(folder+'/test.csv', dinos[0], obstacles, game_speed, 1, screen)
        pygame.display.update()
        clock.tick(game_speed)
        dinos[0].steps += 1
        #steps = steps + 1
        #aumentar la velocidad segun los puntos
        #game_speed = game_speed +0.05


init()
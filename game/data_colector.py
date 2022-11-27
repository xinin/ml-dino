import pygame
import csv
from pathlib import Path
from csv import writer

class DataCollector:

    header = ['distance_next', 'x_next', 'y_next', 'width_next', 'height_next', 'y_dino', 'game_speed']

    def write_data(filename, dino, obstacles, game_speed, action, SCREEN):
        
        obs = sorted(obstacles, key=lambda x: x.rect.x)    
     
        for obstacle in obs:
           if obstacle.rect.x >= dino.rect.x:

                file = None
                if not Path(filename).is_file():
                    with open(filename, 'w') as file:
                        writer_obj = writer(file)
                        writer_obj.writerow(DataCollector.header)
                        file.close()
                else:
                    with open(filename, 'a') as file:
                        writer_obj = writer(file)
                        writer_obj.writerow(
                            [
                                abs(dino.rect.x - obstacle.rect.x),
                                obstacle.rect.x,
                                obstacle.rect.y,
                                obstacle.rect.width,
                                obstacle.rect.height,
                                dino.rect.y,
                                game_speed,
                                action
                            ]
                        )
                        file.close()

                pygame.draw.rect(SCREEN, (255,0,0), pygame.Rect(obstacle.rect.x, obstacle.rect.y, obstacle.rect.width, obstacle.rect.height),  5, 5)
                break
        




        
import pygame
import csv
from pathlib import Path
from csv import writer
import os

class DataCollector:

    #header = ['distance_next', 'x_next', 'y_next', 'width_next', 'height_next', 'y_dino', 'game_speed', 'action']
    header = ['distance_next', 'y_next', 'width_next', 'height_next', 'y_dino', 'game_speed', 'action']

    def write_data(filename, dino, obstacles, game_speed, action):
        
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
                                #obstacle.rect.x,
                                obstacle.rect.y,
                                obstacle.rect.width,
                                obstacle.rect.height,
                                dino.rect.y,
                                game_speed,
                                action
                            ]
                        )
                        file.close()

                #pygame.draw.rect(SCREEN, (255,0,0), pygame.Rect(obstacle.rect.x, obstacle.rect.y, obstacle.rect.width, obstacle.rect.height),  5, 5)
                break
        
    def delete_last_action(folder, index):
        filename = folder+'/dino_'+str(index)+'.csv'
        with open(filename, "r+") as f:
            current_position = previous_position = f.tell()
            while f.readline():
                previous_position = current_position
                current_position = f.tell()
            f.truncate(previous_position)

    def rename_file(folder, index, dino_steps):
        os.rename(folder+'/dino_'+str(index)+'.csv', folder+'/'+str(dino_steps)+'_dino_'+str(index)+'.csv')



        
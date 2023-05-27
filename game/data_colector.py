from pathlib import Path
from csv import writer, reader
import os

class DataCollector:

    header = ['distance_next', 'y_next', 'width_next', 'height_next', 'y_dino', 'game_speed', 'action', 'score', 'last_failed']

    def write_data(filename, dino, obstacles, game_speed, action, score):
        
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
                                obstacle.rect.y,
                                obstacle.rect.width,
                                obstacle.rect.height,
                                dino.rect.y,
                                game_speed,
                                action,
                                score,
                                int(dino.last_failed)
                            ]
                        )
                        file.close()
                
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

    def delete_failures(score_file, training_file):

        rows_to_skip = 2
        
        folder = training_file[:training_file.rfind('/')]
        if not Path(folder).is_dir():
            os.mkdir(folder)
            
        print("score_file", score_file)
        print("training_file", training_file)
        
        with open(score_file, 'r') as input_file, open(training_file, 'w') as output_file:
            csv_reader = reader(input_file)
            csv_writer = writer(output_file)
            #rows_to_skip = 0
            previous_rows = []
            for row in csv_reader:
                print("row", row)
                #if rows_to_skip > 0:
                #    # Skip this row and decrement the rows_to_skip counter
                #    rows_to_skip -= 1
                #    continue
                if row[-1] == '1':
                    # Skip this row and the three previous rows
                    #rows_to_skip = 4
                    previous_rows.clear()
                    continue
                previous_rows.append(row)
                if len(previous_rows) > rows_to_skip:
                    csv_writer.writerow(previous_rows.pop(0))
        
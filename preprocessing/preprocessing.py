import os
import numpy as np
import cv2




class Preprocessing():
    def __init__(self):
        pass

    @staticmethod    
    def tag_data(file_path):

        
        for subdir, dirs, files in os.walk(file_path):
            for file in files:
                
                img = cv2.imread(os.path.join(subdir, file))
                print(img.shape)
            
            # for directory in subdir:
            #     print(directory)
            print("-"*10)

        return 1, 2
        


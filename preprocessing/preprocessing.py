import os
import numpy as np
import cv2




class Preprocessing():
    def __init__(self):
        pass

    @staticmethod    
    def tag_data(file_path):

        X = []
        y = []
        for subdir, dirs, files in os.walk(file_path):
            for file in files:
                img = cv2.imread(os.path.join(subdir, file))
                img = cv2.resize(img, (32,32), cv2.INTER_CUBIC)
                if "Die" in file:
                    X.append(img)
                    y.append(0)
                elif "duck" in file:
                    X.append(img)
                    y.append(2)
                    
            
            # for directory in subdir:
            #     print(directory)
            print("-"*10)

        return np.array(X), np.array(y)
        


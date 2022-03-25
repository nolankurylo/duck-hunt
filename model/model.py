import numpy as np
import tensorflow as tf
import time
import cv2
import os

class Model:
    def __init__(self):
        self.prev_detections = []
        self.predict = tf.saved_model.load("./inference/saved_model")
        self.test_prediction()
    
    def make_prediction(self, current_frame):
        """ Makes a prediction using the saved model fine tuned from SSD Mobilenet
        :param current_frame: (1024,768,3) RGB image
        :return detections dict
        """
        
        input_tensor = tf.convert_to_tensor(current_frame)[tf.newaxis, ...]
        start_time = time.time()
        detections = self.predict(input_tensor)
        end_time = time.time()
        print(f"Total inference time {end_time - start_time}")
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        return detections

    def test_prediction(self):
        """ Initializes the model
        :return None
        """
        test_image = np.ones(shape=(1024,768,3))
        input_tensor = tf.convert_to_tensor(test_image, tf.uint8)[tf.newaxis, ...]
        detections = self.predict(input_tensor)
        print("MODEL READY!!!")
    
    def get_shot_prediction(self,curr_detections):
        shoot_coords = []
        if not self.prev_detections:
            return shoot_coords
        print('prev points')
        print(self.prev_detections)
        print('current points')
        print(curr_detections)
        
        for prev_duck in self.prev_detections:
            #find closest duck
            min_dist = None
            min_coords = ()
            for curr_duck in curr_detections:
                new_dist = np.sqrt((prev_duck[0] - curr_duck[0])**2 + (prev_duck[1] - curr_duck[1])**2)
                if (min_dist is None):
                    min_dist = new_dist
                    min_coords = (curr_duck[0],curr_duck[1])
                elif (new_dist < min_dist):
                    min_dist = prev_dist
                    min_coords = (curr_duck[0],curr_duck[1])
            
            if min_dist > : 
                return shoot_coords
            #find dx and dy
            dy =  min_coords[0] - prev_duck[0]
            dx =  min_coords[1] - prev_duck[1]
            # prediction strength
            p = 1
            shoot_coords.append((min_coords[0]+p*dy,min_coords[1]+p*dx))
        return shoot_coords
            
                
#     #First find closest duck
#       euclidean_distance = np.sqrt((prev_duck))

import numpy as np
import tensorflow as tf
import time
import cv2
import os

class Model:
    def __init__(self):
        self.prev_detections = []
        self.predict = tf.saved_model.load("./trained_models/mobilenet/saved_model")
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
        # print(f"Total inference time {end_time - start_time}")
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
        """ Matches the closest duck from the previous frame to determine how much distance elapsed,
        add that amount of distane in the direction it was going to "guess" where it will be in the next frame
        :param curr_detections: current frames's detections
        :return shoot_coords dict
        """
        shoot_coords = []
        if not self.prev_detections:
            return shoot_coords
        
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
                    min_dist = new_dist
                    min_coords = (curr_duck[0],curr_duck[1])
            #find dx and dy
            dy =  min_coords[0] - prev_duck[0]
            dx =  min_coords[1] - prev_duck[1]
            # prediction strength
            p = 1
            coordinate = (min_coords[0]+p*dy,min_coords[1]+p*dx)
            shoot_coords.append({'coordinate': coordinate, 'move_type': 'absolute'})
        return shoot_coords
            
                
    def get_valid_duck_coords(self,scores,boxes,frame):
        """ Returns list of (x,y) coordinates from current frame detected by the model above a specified threshold
        :param scores: current frames's detection scores
        :param boxes: current frames's detection boxes
        :param frame: current frame
        :return coords list of (x,y) tuples
        """
        threshold = 0.3
        coords = []
        for i in range(0,len(scores)):

            curr_box = boxes[i]
            if scores[i] > threshold:          
                x1 = int(frame.shape[1]*boxes[i][1])
                x2 = int(frame.shape[1]*boxes[i][3])
                y1 = int(frame.shape[0]*boxes[i][0])
                y2 = int(frame.shape[0]*boxes[i][2])
                coords.append((((y1+y2)//2),((x1+x2)//2)))
        return coords
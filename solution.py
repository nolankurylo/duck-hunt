import time
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from model import Model
from utils import Utils

model = Model()
utils = Utils()


"""
Replace following with your own algorithm logic

Two random coordinat
# min_
# def predictnext_coords()prev_duck,current_ducks:
#     #First find closest duck
    
"""


def GetLocation(move_type, env, current_frame):

    time.sleep(1)  # artificial one second processing time

    detections = model.make_prediction(current_frame)
    

    utils.plot_frame(current_frame.copy(), detections.copy())

    scores = np.array(detections['detection_scores'])
    boxes = np.array(detections['detection_boxes'])

    max_score_idx = np.argmax(scores)

    shoot_box = boxes[max_score_idx]
    shoot_score = scores[max_score_idx]
    
    if(shoot_score < 0.3):
        print('no ducks')
        return [{'coordinate': 8, 'move_type': 'relative'}]

    x1 = int(current_frame.shape[1]*shoot_box[1])
    x2 = int(current_frame.shape[1]*shoot_box[3])
    y1 = int(current_frame.shape[0]*shoot_box[0])
    y2 = int(current_frame.shape[0]*shoot_box[2])

    shoot_x = ((x1+x2)//2)
    shoot_y = ((y1+y2)//2)

    new_coords = model.get_shot_prediction([(shoot_y,shoot_x)])
    print('predicted shot: ')
    print(new_coords)





    # Use relative coordinates to the current position of the "gun", defined as an integer below
    if move_type == "relative":
        """
        North = 0
        North-East = 1
        East = 2
        South-East = 3
        South = 4
        South-West = 5
        West = 6
        North-West = 7
        NOOP = 8
        """
        coordinate = env.action_space.sample()
        # Use absolute coordinates for the position of the "gun", coordinate space are defined below
    else:
        """
        (x,y) coordinates
        Upper left = (0,0)
        Bottom right = (W, H) 
        """
        # coordinate = env.action_space_abs.sample()
        model.prev_detections = [(shoot_y,shoot_x)]
        if new_coords:
            coordinate = new_coords[0]
        else:
            return [{'coordinate': 8, 'move_type': 'relative'}]
        

    return [{'coordinate': coordinate, 'move_type': move_type}]

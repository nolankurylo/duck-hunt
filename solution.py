import time
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from modelUtils import ModelUtils
from plotter import Plotter

MODEL_TYPE = 'yolo' # resnet | mobilenet | yolo
USE_FRAME_DISPLACE_ALGO = False

model = ModelUtils(model_type=MODEL_TYPE)
plotter = Plotter()


"""
Replace following with your own algorithm logic
"""

def GetLocation(move_type, env, current_frame):

    time.sleep(1)  # artificial one second processing time

    if MODEL_TYPE == 'yolo':
        current_ducks = model.yolo_predict(current_frame)
    else:
        detections = model.ssd_predict(current_frame)
        scores = np.array(detections['detection_scores'])
        boxes = np.array(detections['detection_boxes'])
        classes = np.array(detections['detection_classes'])

        scores = scores[np.where(classes == 2)]
        boxes = boxes[np.where(classes == 2)]    

        current_ducks = model.get_valid_duck_coords(scores,boxes,current_frame)

    if len(current_ducks) == 0:
        model.prev_detections = current_ducks
        return [{'coordinate': 8, 'move_type': 'relative'}]

    if USE_FRAME_DISPLACE_ALGO:
        new_coords = model.get_shot_prediction(current_ducks)
    else:
        new_coords = []
        for duck in current_ducks:
            new_coords.append({'coordinate': duck, 'move_type': 'absolute'})

    

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
        return [{'coordinate': coordinate, 'move_type': move_type}]
        
        # Use absolute coordinates for the position of the "gun", coordinate space are defined below
    else:
        """
        (x,y) coordinates
        Upper left = (0,0)
        Bottom right = (W, H) 
        """
        model.prev_detections = current_ducks
        
        if len(new_coords) > 0:
            return new_coords
        else:
            return [{'coordinate': 8, 'move_type': 'relative'}]
        


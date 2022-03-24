import time
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


import shutil
if os.path.exists("./fig"):
    shutil.rmtree("./fig")
os.mkdir("fig")

"""
Replace following with your own algorithm logic

Two random coordinate generator has been provided for testing purposes.
Manual mode where you can use your mouse as also been added for testing purposes.
"""
model = tf.saved_model.load("./inference/saved_model")


def detect_and_classify_udder(image, detections):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    scores = np.array(detections['detection_scores'])

    boxes = detections['detection_boxes']
    # add a threshold under which the box is not drawn
    real_boxes = boxes[np.where(scores > 0.3)]
    real_scores = scores[np.where(scores > 0.3)]

    real_boxes_scaled = []

    for box in real_boxes:
        x1 = int(image.shape[1]*box[1])
        x2 = int(image.shape[1]*box[3])
        y1 = int(image.shape[0]*box[0])
        y2 = int(image.shape[0]*box[2])

        real_boxes_scaled.append([x1, x2, y1, y2])

    duck_found = False
    x1, x2, y1, y2 = 0, 0, 0, 0
    for box, score in zip(real_boxes_scaled, real_scores):
        duck_found = True
        x1, x2, y1, y2 = box
        # draw the rectangle with the coordinates

        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 5)
        # add caption with putText

        image = cv2.putText(image, str(score), (x1-22, y1-16),
                            0, 0.8, (255, 255, 0), 2)
        break
    if duck_found:
        plt.figure(figsize=(8, 8))
        plt.title(f"x1:{x1},x2:{x2},y1:{y1},y2:{y2}")
        # image = cv2.transpose(image)
        # image = cv2.flip(image, flipCode=1)
        plt.imshow(image)
        plt.savefig(f"./fig/{time.time()}.png")


def GetLocation(move_type, env, current_frame):

    time.sleep(1)  # artificial one second processing time

    input_tensor = tf.convert_to_tensor(current_frame)[tf.newaxis, ...]
    start_time = time.time()
    detections = model(input_tensor)
    end_time = time.time()
    print(f"Total inference time {end_time - start_time}")
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    # detect_and_classify_udder(current_frame.copy(), detections.copy())

    scores = np.array(detections['detection_scores'])
    boxes = np.array(detections['detection_boxes'])

    max_score_idx = np.argmax(scores)

    shoot_box = boxes[max_score_idx]

    x1 = int(current_frame.shape[1]*shoot_box[1])
    x2 = int(current_frame.shape[1]*shoot_box[3])
    y1 = int(current_frame.shape[0]*shoot_box[0])
    y2 = int(current_frame.shape[0]*shoot_box[2])

    shoot_score = scores[max_score_idx]

    if(shoot_score < 0.2):
        print(f"NOOP - score: {shoot_score}")
        return [{'coordinate': 8, 'move_type': 'relative'}]

    shoot_x = ((x1+x2)//2)
    shoot_y = ((y1+y2)//2)

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

        coordinate = (shoot_y, shoot_x)
        print(f"COORDINATE SET - {coordinate}")

    return [{'coordinate': coordinate, 'move_type': move_type}]

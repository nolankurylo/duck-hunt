import time
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


class Utils:

    def __init__(self):
        if os.path.exists("./fig"):
            shutil.rmtree("./fig")
        os.mkdir("fig")

    @staticmethod
    def plot_frame(image, detections):
        """ Plots bounding boxes on all ducks found in current frame
        :param detections: dict predicted by fine tuned model
        :return None
        """
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scores = np.array(detections['detection_scores'])

        boxes = detections['detection_boxes']

        real_boxes = boxes[np.where(scores > 0.3)] # threshold bad predictions
        real_scores = scores[np.where(scores > 0.3)]

        real_boxes_scaled = []

        for box in real_boxes:
            x1 = int(image.shape[1]*box[1])
            x2 = int(image.shape[1]*box[3])
            y1 = int(image.shape[0]*box[0])
            y2 = int(image.shape[0]*box[2])

            real_boxes_scaled.append([x1, x2, y1, y2])

        for box, score in zip(real_boxes_scaled, real_scores):
            x1, x2, y1, y2 = box
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 5)
            image = cv2.putText(image, str(score), (x1-22, y1-16),
                                0, 0.8, (255, 255, 0), 2)
        
        plt.figure(figsize=(8, 8))
        plt.title(f"Bounding Box predictions")
        plt.imshow(image)
        plt.savefig(f"./fig/{time.time()}.png")
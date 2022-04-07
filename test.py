import os
import torch
import numpy as np

test_image = np.ones(shape=(1024,768,3))
predict = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

res = predict(test_image)
result = res.pandas().xyxy
print(result)
print(len(result))


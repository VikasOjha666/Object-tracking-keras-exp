from keras.models import load_model
import numpy as np
import cv2
from yolo_utils import *
import matplotlib.pyplot as plt

model=load_model('tiny_yolo.h5')




def detectObjects(image):
    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)

    # netout = new_model.predict(input_image)
    netout = model.predict(input_image)
    boxes,scores,classes = interpret_netout(image, netout[0])

    return boxes,scores,classes


# image=cv2.imread('./man.jpg')
# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# boxes,scores,classes=detectObjects(image)

# print(boxes)
# print(scores)
# print(classes)








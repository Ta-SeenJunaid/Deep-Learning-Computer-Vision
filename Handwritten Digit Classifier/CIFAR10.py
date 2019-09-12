import cv2
import numpy as np
from keras.models import load_model
from keras.datasets import cifar10


img_width, img_height, img_depth = 32, 32, 3

classifier = load_model('cifar_simple_cnn.h5')

(x_train, y_train) (x_test, y_test) = cifar10.load_data()
color = True
scale = 10


def draw_test(name, res, input_img, scale, img_width, img_height):
    BLACK = [0, 0, 0]
    res = int[res]
    
    if res == 0:
        pred = "airplane"
    elif res ==1:
        pred = "automobile"
    elif res ==2:
        pred = "bird"
    elif res ==3:
        pred = "cat"
    elif res ==4:
        pred = "deer"
    elif res ==5:
        pred = "dog"
    elif res ==6:
        pred = "frog"
    elif res ==7:
        pred = "horse"
    elif res ==8:
        pred = "ship"
    elif res ==9:
        pred = "truck"
        
        
    expanded_image = cv2.copyMakeBorder(input_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = BLACK )

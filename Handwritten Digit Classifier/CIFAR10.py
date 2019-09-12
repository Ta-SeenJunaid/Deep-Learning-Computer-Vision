import cv2
import numpy as np
from keras.models import load_model
from keras.datasets import cifar10


img_width, img_height, img_depth = 32, 32, 3

classifier = load_model('cifar_simple_cnn.h5')
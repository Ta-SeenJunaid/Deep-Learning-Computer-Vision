import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import load_model

classifier = load_model('mnist_simple_cnn.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def draw_test_image(name, pred, input_img):
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(input_img, 0, 0, 0, imageL.shape[0],
                                        cv2.BORDER_CONSTANT, value=BLACK)
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (152, 70), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (255, 0, 0), 2)
    cv2.imshow(name, expanded_image)
    
for i in range(0, 12):
    rand = np.random.randint(0,len(x_test))
    input_img = x_test[rand]

    imageL = cv2.resize(input_img, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC) 
    input_img = input_img.reshape(1,28,28,1)
    
    res = str(classifier.predict_classes(input_img, 1, verbose = 0)[0])
    draw_test_image("Prediction", res, imageL) 
    cv2.waitKey(0)

cv2.destroyAllWindows()



    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: okokprojects
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        
        # load model
        model = load_model('model_eye.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (300, 300))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        classes = np.argmax(result)
        print("", classes)

        if classes <=0:
            prediction = 'User Uploaded image Bulging eyes'
            print("Classification result", prediction)
        elif classes <=1:
            prediction = 'User Uploaded image Cataracts'
            print("Classification result", prediction)
        elif classes <=2:
            prediction = 'User Uploaded image Crossed_Eyes'
            print("Classification result", prediction)
        elif classes <=3:
            prediction = 'User Uploaded image Glaucoma'
            print("Classification result", prediction)
        elif classes <=4:
            prediction = 'User Uploaded image Normal'
            print("Classification result", prediction)
        elif classes <=5:
            prediction = 'User Uploaded image Uveitis'
            print("Classification result", prediction)

        else:
            prediction = 'User Uploaded image Uveitis'
            print("Classification result", prediction)

        return [prediction]

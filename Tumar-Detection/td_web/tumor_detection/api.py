from __future__ import division, print_function

# coding=utf-8
import numpy as np
# Keras
from tensorflow.keras.preprocessing import image


# Flask utils


class TumorDetection:
    def __init__(self, ml_model):
        self.model = ml_model

    def predict(self, image_path):
        img = image.load_img(image_path, target_size=(256, 256))
        # Preprocessing the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Be careful how your trained model deals with the input
        # otherwise, it won't make correct prediction!
        # x = preprocess_input(x, mode='caffe')

        preds = self.model.predict(x)
        return preds

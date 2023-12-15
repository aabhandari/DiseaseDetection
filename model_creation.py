import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception, DenseNet169
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def create_models():
    img_size = (150, 150)
    num_classes = ...  # Defining the number of classes based on  dataset

    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    densenet_base = DenseNet169(weights='imagenet', include_top=False, input_shape=(*img_size, 3))

    def fine_tune_model(base_model):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        return model

    xception_model = fine_tune_model(xception_base)
    densenet_model = fine_tune_model(densenet_base)

    return [xception_model, densenet_model]

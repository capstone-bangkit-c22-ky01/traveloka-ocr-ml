import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

from modules.feature_extraction import VGG_FeatureExtractor


class Model(keras.models.Model):
    def __init__(self, opt, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.opt = opt

        print("No Transformation Module")

        self.FeatureExtraction = VGG_FeatureExtractor(opt.output_channel)
        self.FeatureExtraction_output = opt.output_channel
        # untuk sekarang
        self.AdaptiveAvgPool = tfa.layers.AdaptiveAveragePooling2D(output_size=(24, 1))
        print("No sequence modelling module specified")
        self.SequenModelling_output = self.FeatureExtraction_output

        self.Prediction = layers.Dense(opt.num_class)

    def call(self, X, text, training=None):

        """Feature Extraction Stage"""
        visual_feature = self.FeatureExtraction(X)
        visual_feature = self.AdaptiveAvgPool(
            tf.transpose(visual_feature, perm=[0, 2, 1, 3])
        )  # [b, w, h, c] -> [b, h, w, c]
        visual_feature = tf.squeeze(visual_feature, axis=2)

        contextual_feature = visual_feature

        prediction = self.Prediction(contextual_feature)
        return prediction

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

from modules.feature_extraction import VGG_FeatureExtractor

tf.keras.backend.set_image_data_format("channels_first")


class Model(keras.models.Model):
    def __init__(self, opt, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.opt = opt

        print("No Transformation Module")

        self.FeatureExtraction = VGG_FeatureExtractor(opt.output_channel)
        self.FeatureExtraction_output = opt.output_channel
        # untuk sekarang
        self.AdaptiveAvgPool = tfa.layers.AdaptiveAveragePooling2D(output_size=(1))
        opt.num_class = len(opt.character)
        print("No sequence modelling module specified")
        self.SequenModelling_output = self.FeatureExtraction_output

        self.Prediction = layers.Dense(opt.num_class)

    def call(self, X, training=None):

        """Feature Extraction Stage"""
        visual_feature = self.FeatureExtraction(X)
        visual_feature = self.AdaptiveAvgPool(
            tf.transpose(visual_feature, perm=[0, 3, 1, 2])
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = tf.squeeze(visual_feature, axis=3)

        contextual_feature = visual_feature

        prediction = self.Prediction(contextual_feature)

        return prediction

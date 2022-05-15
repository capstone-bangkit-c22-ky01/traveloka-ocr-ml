import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AttentionCell(keras.models.Model):
    
    def __init__(self, hidden_size, num_embeddings):
        super().__init__()
        self.i2h = layers.Dense(hidden_size, use_bias=False)
        self.h2h = layers.Dense(hidden_size)
        self.score = layers.Dense(1, use_bias=False)
        self.rnn = layers.LSTMCell()
        self.hidden_size = hidden_size
        
    def call(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = tf.expand_dims(self.h2h(prev_hidden[0]), axis=1)
        e = self.score(tf.math.tanh(tf.add(batch_H_proj + prev_hidden_proj))) # rawan error
        
        alpha = tf.math.softmax(e, axis=1)
        context = tf.squeeze(tf.matmul(tf.transpose(alpha, [0, 2, 1]), batch_H), axis=1)
        concat_context = tf.concat([context, char_onehots], axis=1)
        cur_hidden = self.rnn(concat_context, prev_hidden) # rawan error
        return cur_hidden, alpha

class Attention(keras.models.Model):
    
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.attention_cell = AttentionCell()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = layers.Dense(num_classes)
        
    @staticmethod    
    def _char_to_onehot(input_char, onehot_dim=38):
        input_char = tf.expand_dims(input_char, axis=1)
        one_hot = tf.one_hot(input_char, onehot_dim)
        return one_hot # rawan error
    
    def call(self, batch_H, text, is_train=True, batch_max_length=25):
        return None
        

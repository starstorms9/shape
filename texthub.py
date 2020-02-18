#%% Imports
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Reshape, BatchNormalization, GRU, LSTM, Bidirectional, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras import regularizers
import numpy as np
import os

#%% RNN Class
class TextHub(tf.keras.Model):
    def __init__(self, latent_dim, learning_rate=6e-4, training=True):
        super(TextHub, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.training = training
        self.dropoutRate = 0.2
        self.hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2", input_shape=[], dtype=tf.string)
        self.hub_layer.trainable = False
        model = tf.keras.Sequential()
        model.add(self.hub_layer)
        model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(self.dropoutRate))
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(self.dropoutRate))
        model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(self.dropoutRate))
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(self.dropoutRate))
        model.add(tf.keras.layers.Dense(latent_dim, activation='linear'))      
        self.model = model
    
    def __call__(self, text) :
        return self.model(text)
    
    def sample(self, x):
        ypred = self.model(x, training=False)
        self.model.reset_states()
        return ypred
    
    @tf.function
    def compute_loss(self, ypred, y):
        return tf.keras.losses.mean_squared_error(ypred, y)
    
    @tf.function
    def trainStep(self, x, y) :
        with tf.GradientTape() as tape:
            ypred = self.model(x, training=True)
            loss = self.compute_loss(ypred, y)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))  
        return loss
    
    def saveMyModel(self, dir_path, epoch):        
        self.model.save_weights(os.path.join(dir_path, 'epoch_{}.h5'.format(epoch)))
        
    def loadMyModel(self, dir_path, epoch):        
        self.model.load_weights(os.path.join(dir_path, 'epoch_{}.h5'.format(epoch)))
        
    def setLR(self, learning_rate) :
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def printMSums(self) :
        print("Text Embed Net Summary:\n")
        self.model.summary()
        
    def printIO(self) :
        print("\nText Embed Net Summary (input then output):")
        print(self.model.input_shape)
        print(self.model.output_shape)
        print("\nPer Layer:")
        for i, layer in enumerate(self.model.layers) :
            print('{:2d} : {:20} In: {:20} Out: {}'.format(i, layer.name, repr(layer.input_shape), layer.output_shape))


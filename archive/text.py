#%% Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Reshape, BatchNormalization, GRU, LSTM, Bidirectional, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras import regularizers
import numpy as np
import os

#%% RNN Class
class Text(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, batch_size, latent_dim, learning_rate=6e-4, training=True):
        super(Text, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.text_model = self.makeGRUModel(vocab_size, embed_dim, latent_dim, 0.15, batch_size)
        # self.text_model = self.makeLSTMModel(vocab_size, embed_dim, latent_dim, 0.1)
        self.training = training
    
    def __call__(self) :
        pass
    
    def makeGRUModel(self, vocab_size, embed_dim, latent_dim, dropout_rate, batch_size) :
        model = tf.keras.Sequential()
        model.add(Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))
        model.add(SpatialDropout1D(dropout_rate))
        model.add(Bidirectional(GRU(64, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')))
        model.add(Bidirectional(GRU(128, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')))
        model.add(Bidirectional(GRU(256, stateful=True, recurrent_initializer='glorot_uniform')))
        model.add(Dropout(dropout_rate))
        model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(latent_dim, activation='linear'))
        return model
    
    def makeLSTMModelNew(self, vocab_size, embed_dim, latent_dim, dropout_rate) :
        model = tf.keras.Sequential()
        model.add(Embedding(vocab_size, embed_dim))
        model.add(SpatialDropout1D(dropout_rate))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(256)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(latent_dim, activation='linear'))   
        return model
    
    def makeLSTMModel(self, vocab_size, embed_dim, latent_dim, dropout_rate) :
        model = tf.keras.Sequential()
        model.add(Embedding(vocab_size, embed_dim))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(256)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(latent_dim, activation='linear'))
        return model
    
    def sample(self, x):
        ypred = self.text_model(x, training=False)
        self.text_model.reset_states()
        return ypred
    
    @tf.function
    def compute_loss(self, ypred, y):
        return tf.keras.losses.mean_squared_error(ypred, y)
    
    @tf.function
    def trainStep(self, x, y) :
        with tf.GradientTape() as tape:
            ypred = self.text_model(x, training=True)
            loss = self.compute_loss(ypred, y)
        gradients = tape.gradient(loss, self.text_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.text_model.trainable_variables))  
        return loss
    
    def saveMyModel(self, dir_path, epoch):        
        self.text_model.save_weights(os.path.join(dir_path, 'epoch_{}.h5'.format(epoch)))
        
    def loadMyModel(self, dir_path, epoch):        
        self.text_model.load_weights(os.path.join(dir_path, 'epoch_{}.h5'.format(epoch)))
        
    def setLR(self, learning_rate) :
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def printMSums(self) :
        print("Text Embed Net Summary:\n")
        self.text_model.summary()
        
    def printIO(self) :
        print("\nText Embed Net Summary (input then output):")
        print(self.text_model.input_shape)
        print(self.text_model.output_shape)
        print("\nPer Layer:")
        for i, layer in enumerate(self.text_model.layers) :
            print('{:2d} : {:20} In: {:20} Out: {}'.format(i, layer.name, repr(layer.input_shape), layer.output_shape))


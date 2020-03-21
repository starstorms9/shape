'''
This file is the text encoder model.
It uses spacy word embeddings (which are based on GloVe). The medium sized embeddings are used here ('en_core_web_md').

There are 3 LSTM layers and 4 dense layers in this model and a significant amount of regularization.
'''

#%% Imports
import tensorflow as tf
import spacy
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Reshape, BatchNormalization, GRU, LSTM, Bidirectional, Embedding, Dropout, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
from tensorflow.keras import regularizers
import numpy as np
import os

#%% RNN + Dense Class that extends the standard keras model
class TextSpacy(tf.keras.Model):
    spacy_embeddings = None
    
    def __init__(self, latent_dim, learning_rate=6e-4, max_length=100, training=True, embeddings=[]):
        super(TextSpacy, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.training = training
        self.dropoutRate = 0.2
        
        spacy_embeddings = self.get_embeddings() if len(embeddings)==0 else embeddings
        
        model = tf.keras.Sequential()
        model.add(Embedding(spacy_embeddings.shape[0], spacy_embeddings.shape[1], input_length=max_length, trainable=False, weights=[spacy_embeddings] ) )
        model.add(SpatialDropout1D(self.dropoutRate))     
        model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))))
        model.add(Dropout(self.dropoutRate))
        model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))))
        model.add(Dropout(self.dropoutRate))
        model.add(Bidirectional(LSTM(256, kernel_regularizer=regularizers.l2(0.001))))
        model.add(Dropout(self.dropoutRate))
        model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(self.dropoutRate))
        model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(self.dropoutRate))
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(self.dropoutRate))
        model.add(Dense(latent_dim, activation='linear'))    
        self.model = model
    
    def __call__(self, text) :
        return self.model(text)
    
    def get_embeddings(self):
        nlp = spacy.load("en_core_web_md", parser=False, tagger=False, entity=False)
        vocab = nlp.vocab
        max_rank = max(lex.rank for lex in vocab if lex.has_vector)
        vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
        for lex in vocab:
            if lex.has_vector:
                vectors[lex.rank] = lex.vector
        return vectors
    
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
            
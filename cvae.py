#%% Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv3D, Conv3DTranspose, InputLayer, Flatten, Reshape
import numpy as np

#%% VAE Class
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, input_dim):
        super(CVAE, self).__init__()
        
        self.optimizer = tf.keras.optimizers.Adam(6e-4)
        self.latent_dim = latent_dim
        self.gen_layers = 3
        self.gen_init_size = int(input_dim / (2 ** (self.gen_layers-1)))
        
        self.inference_net = tf.keras.Sequential()
        self.inference_net.add(InputLayer(input_shape=(input_dim, input_dim, input_dim, 1)))
        self.inference_net.add(Conv3D( filters=16, kernel_size=3, strides=(2, 2, 2), activation='relu'))
        self.inference_net.add(Conv3D( filters=32, kernel_size=3, strides=(2, 2, 2), activation='relu'))
        self.inference_net.add(Flatten())
        self.inference_net.add(Dense(latent_dim + latent_dim))
          
        self.generative_net = tf.keras.Sequential()
        self.generative_net.add(InputLayer(input_shape=(latent_dim,)))
        self.generative_net.add(Dense(units= (self.gen_init_size ** 3) * 8  , activation=tf.nn.relu))
        self.generative_net.add(Reshape(target_shape=(self.gen_init_size, self.gen_init_size, self.gen_init_size, 8)))
        self.generative_net.add(Conv3DTranspose( filters=32, kernel_size=3, strides=(2, 2, 2), padding="SAME", activation='relu'))
        self.generative_net.add(Conv3DTranspose( filters=16, kernel_size=3, strides=(2, 2, 2), padding="SAME", activation='relu'))
        self.generative_net.add(Conv3DTranspose( filters=1,  kernel_size=3, strides=(1, 1, 1), padding="SAME"))

    def reconstruct(self, train_x) :
        mean, logvar = self.encode(train_x)
        z = self.reparameterize(mean, logvar)
        x_logits = self.decode(z)
        probs = tf.sigmoid(x_logits)
        return probs

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(5, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x, reparam=False):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        if (reparam) :
            return self.reparameterize(mean, logvar)
        return mean, logvar
        
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def printMSums(self) :
        print("Inference Net Summary:\n")
        self.inference_net.summary()
        print("\nGenerative Net Summary:\n")
        self.generative_net.summary()
        
    def setLR(self, learning_rate) :
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis )
    
    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
    
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3, 4])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    @tf.function
    def compute_apply_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
    def saveWeights(self, filepath, suffix) :
        self.inference_net.save_weights(filepath+'/inf{}.h5'.format(suffix))
        self.generative_net.save_weights(filepath+'/gen{}.h5'.format(suffix))
    
    def loadWeights(self, filepath, suffix='') :
        self.inference_net.load_weights(filepath+'/inf{}.h5'.format(suffix))
        self.generative_net.load_weights(filepath+'/gen{}.h5'.format(suffix))
        
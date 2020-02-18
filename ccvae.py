#%% Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv3D, Conv3DTranspose, InputLayer, Input, Flatten, Reshape, BatchNormalization, Embedding, Concatenate
import numpy as np
import utils as ut

#%% VAE Class
class CCVAE(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, learning_rate, num_classes, embed_dim):
        super(CCVAE, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.latent_dim = latent_dim
        self.gen_layers = 5
        self.gen_init_size = int(input_dim / (2 ** (self.gen_layers-1)))
        self.reshape_channels = 16
        self.prior_dim = self.gen_init_size ** 3 * self.reshape_channels
                        
        self.enc_model = tf.keras.Sequential()
        self.enc_model.add(InputLayer(input_shape=(input_dim, input_dim, input_dim, 1)))
        self.enc_model.add(Conv3D( filters=32, kernel_size=4, strides=(2, 2, 2), padding="SAME", activation='relu'))
        self.enc_model.add(Conv3D( filters=64, kernel_size=4, strides=(2, 2, 2), padding="SAME", activation='relu'))
        self.enc_model.add(Conv3D( filters=128,kernel_size=4, strides=(2, 2, 2), padding="SAME", activation='relu'))
        self.enc_model.add(Flatten())
        self.enc_model.add(Dense(self.latent_dim + self.latent_dim))
        
        txt_input = Input(shape=(1,))
        xt = Embedding(input_length=1, input_dim=num_classes, output_dim=embed_dim)(txt_input)
        xt = Flatten()(xt)
        xt = Dense( int(self.prior_dim/2), activation=tf.nn.relu )(xt)
        xt = Dense( self.prior_dim, activation=tf.nn.relu )(xt)
        txt_output = Reshape(target_shape=(self.gen_init_size, self.gen_init_size, self.gen_init_size, self.reshape_channels))(xt)
        
        enc_input = Input(shape=(self.latent_dim,))
        xgen = Dense(units= (self.gen_init_size ** 3) * self.reshape_channels, activation=tf.nn.relu)(enc_input)
        xgen = Reshape(target_shape=(self.gen_init_size, self.gen_init_size, self.gen_init_size, self.reshape_channels))(xgen)        
        xgen = Concatenate(axis=-1)([txt_output, xgen])                
        xgen = Conv3DTranspose( filters=256, kernel_size=4, strides=(2, 2, 2), padding="SAME", activation='relu')(xgen)
        xgen = Conv3DTranspose( filters=128, kernel_size=4, strides=(2, 2, 2), padding="SAME", activation='relu')(xgen)
        xgen = Conv3DTranspose( filters=64,  kernel_size=4, strides=(2, 2, 2), padding="SAME", activation='relu')(xgen)
        xgen = Conv3DTranspose( filters=32,  kernel_size=4, strides=(2, 2, 2), padding="SAME", activation='relu')(xgen)
        gen_out = Conv3DTranspose( filters=1,   kernel_size=4, strides=(1, 1, 1), padding="SAME")(xgen)
        self.gen_model = tf.keras.models.Model(inputs=[enc_input, txt_input], outputs=[gen_out])
        
    def reconstruct(self, train_x, catids) :
        if (len(train_x.shape) == 3) :
            tx = (train_x)[None,...,None]
        elif (len(train_x.shape) == 4) :
            tx = (train_x)[None,...]
        else :
            tx = train_x          

        if (type(catids)==int and catids < 100) :
            cats = tf.Variable([catids], tf.int32)
        elif (len(catids.shape)==0) :
            cats = ut.getCats([catids], ut.cf_cat_prefixes)  
        else :
            cats = ut.getCats(catids, ut.cf_cat_prefixes)                       
            
        mean, logvar = self.encode(tx)
        z = self.reparameterize(mean, logvar, randomize=False)
        x_logits = self.decode(z, cats, apply_sigmoid=True)
        if x_logits.shape[0] == 1:
            return x_logits.numpy()[0,:,:,:,0]
        else :
            return x_logits.numpy()[:,:,:,:,0]

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(5, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    # @tf.function
    def encode(self, x, reparam=False):
        mean, logvar = tf.split(self.enc_model(x), num_or_size_splits=2, axis=-1)
        if (reparam) :
            return self.reparameterize(mean, logvar)
        return mean, logvar
        
    # @tf.function
    def reparameterize(self, mean, logvar, randomize=True):
        eps = tf.random.normal(shape=mean.shape)
        if randomize : return eps * tf.exp(logvar * .5) + mean
        else : return tf.exp(logvar * .5) + mean

    # @tf.function
    def decode(self, z, t, apply_sigmoid=False):
        logits = self.gen_model([z,t])
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    @tf.function
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
    
    @tf.function
    def compute_loss(self, x, t) :
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, t)
    
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3, 4])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    @tf.function
    def trainStep(self, x, t) :
        with tf.GradientTape() as tape :
            loss = self.compute_loss(x, t)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))    
        return loss

    def printMSums(self) :
        print("Inference Net Summary:\n")
        self.enc_model.summary()
        print("\nGenerative Net Summary:\n")
        self.gen_model.summary()
        
    def printGenGraph(self) :        
        model = self.gen_model
        tf.keras.utils.plot_model(model.gen_model, show_shapes=True)
        
    def printIO(self) :
        print("\nInference Net Summary (input then output):")
        print(self.enc_model.input_shape)
        print(self.enc_model.output_shape)
        print("\nGenerative Net Summary:")
        print(self.gen_model.input_shape)
        print(self.gen_model.output_shape)
        
    def setLR(self, learning_rate) :
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
#%% Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv3D, Conv3DTranspose, InputLayer, Flatten, Reshape, Concatenate, LeakyReLU, Dropout, BatchNormalization
import numpy as np
import utils as ut

#%% VAE Class
class GAN(tf.keras.Model):
    def __init__(self, latent_dim, vox_size, gen_lr, dis_lr):
        super(GAN, self).__init__()
        
        self.optimizer = tf.keras.optimizers.Adam(6e-4)
        self.latent_dim = latent_dim
        self.gen_layers = 5
        self.gen_init_size = int(vox_size / (2 ** (self.gen_layers-1)))
        
        self.vox_size = vox_size
        self.gen_model = self.make_generator_model()
        self.dis_model = self.make_discriminator_model()        
        self.gen_opt = tf.keras.optimizers.Adam(gen_lr)
        self.dis_opt = tf.keras.optimizers.Adam(dis_lr)        
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
    def make_generator_model(self):
        model = tf.keras.Sequential()
        reshape_start_channels = 256
        reshape_size = 2
        model.add(Dense(reshape_size*reshape_size*reshape_size*reshape_start_channels, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Reshape((reshape_size, reshape_size, reshape_size, reshape_start_channels)))
        model.add(Conv3DTranspose(reshape_start_channels, 4, strides=1, padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
    
        model.add(Conv3DTranspose(128, 4, strides=2, padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv3DTranspose(64, 4, strides=2, padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv3DTranspose(32, 4, strides=2, padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv3DTranspose(1, 4, strides=2, padding='same', use_bias=False, activation='tanh'))
        return model
    
    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(Conv3D(32, 4, strides=2, padding='same', input_shape=[self.vox_size, self.vox_size, self.vox_size, 1]))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv3D(64, 4, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv3D(128, 4, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv3D(256, 4, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Flatten())
        model.add(Dense(1))
        return model
    
    @tf.function
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    @tf.function
    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, voxs, noise):    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
          generated_images = self.gen_model(noise, training=True)
          
          real_output = self.dis_model(voxs, training=True)
          fake_output = self.dis_model(generated_images, training=True)
          
          gen_loss = self.generator_loss(fake_output)
          dis_loss = self.discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, self.gen_model.trainable_variables)
        gradients_of_discriminator = dis_tape.gradient(dis_loss, self.dis_model.trainable_variables)
    
        self.gen_opt.apply_gradients(zip(gradients_of_generator, self.gen_model.trainable_variables))
        self.dis_opt.apply_gradients(zip(gradients_of_discriminator, self.dis_model.trainable_variables))
        
        return gen_loss, dis_loss
    
    @tf.function
    def compute_apply_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))    
               
    def setLR(self, gen_learning_rate, dis_learning_rate) :
        self.gen_opt = tf.keras.optimizers.Adam(gen_learning_rate)
        self.dis_opt = tf.keras.optimizers.Adam(dis_learning_rate)      
    
    def getDiscAcc(self, num_to_test, train_dataset, print_out=False) :
        noise = tf.random.normal([num_to_test, self.latent_dim])
        fake_vox = self.gen_model(noise, training=False)
        real_vox = list(train_dataset.unbatch().shuffle(1000).batch(num_to_test).take(1))[0]
        
        fake_decision = self.dis_model(fake_vox, training=False).numpy()
        real_decision = self.dis_model(real_vox, training=False).numpy()
        fake_acc = np.mean(fake_decision < 0)
        real_acc = np.mean(real_decision > 0)
        
        if (print_out) : 
            print('Percent fakes classified correctly: {:.4f} \nPercent reals classified correctly: {:.4f}'.format(fake_acc, real_acc))
        return fake_acc, real_acc
    
    def showVoxImages(self, epoch, test_input=None, num_examples=0):
        if (num_examples > 0) :
            test_input = tf.random.normal([num_examples, self.latent_dim])
        
        predictions = self.gen_model(test_input, training=False)    
        for i in range(test_input.shape[0]) :
            gen_vox = predictions.numpy()[i,...,0]
            ut.plotVox(gen_vox, title='Epoch {}'.format(epoch), save_fig=True)

    def printMSums(self) :
        print("Generator Model Summary:\n")
        self.gen_model.summary()
        print("\nDiscriminator Model Summary:\n")
        self.dis_model.summary()
        
    def printIO(self) :
        print("\Generator Model Summary (input then output):")
        print(self.gen_model.input_shape)
        print(self.gen_model.output_shape)
        print("\nDiscriminator Model Summary:")
        print(self.dis_model.input_shape)
        print(self.dis_model.output_shape)        
        
    def printLayers(self) :
        print('\nGenerator model layers: ')
        for i, layer in enumerate(self.gen_model.layers) :
            print('{:2d} : {:<15} : {:<25} : {}'.format(i, layer.name, repr(layer.input_shape), layer.output_shape))
        print('\nDiscrinimator model layers: ')
        for i, layer in enumerate(self.dis_model.layers) :
            print('{:2d} : {:<15} : {:<25} : {}'.format(i, layer.name, repr(layer.input_shape), layer.output_shape))    
        

"""
needs tf >=2.1

pixelnormalization:
https://arxiv.org/abs/1710.10196
# from https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/
"""
import pickle
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib

from pylab import plt
from tqdm import trange
from matplotlib.colors import LogNorm
from tensorflow.keras.utils import GeneratorEnqueuer
from tensorflow.keras import backend as K
from functools import partial
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply,UpSampling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets.mnist import load_data

from IPython import display

plotdir='./'
outdir='./'


# load the images into memory
(X_train, y_train), (testX, testy) = load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train,-1)


n_classes = 10
# neural network parameters
n_disc = 5
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
latent_dim = 100
batch_size = 32
plot_format = 'png'

n_samples = len(X_train)
img_shape = (28,28,1)


def generate_real_samples(n_batch):
    """get random sampples and do the last preprocessing on them"""
    while True:

        idx = np.random.randint(0, X_train.shape[0], n_batch)
        batch, batch_cond = X_train[idx], y_train[idx]

        yield [batch, batch_cond]


def generate_latent_points(n_batch):
    # generate points in the latent space and a random condition
    latent = np.random.normal(size=(n_batch, latent_dim))
    # randomly select conditions
    idx = np.random.randint(0, X_train.shape[0], n_batch)
    batch_cond = y_train[idx]

    return [latent, batch_cond]


def generate_latent_points_as_generator(n_batch):
    while True:
        yield generate_latent_points(n_batch)


def generate_fake_samples(n_batch):
    # generate points in latent space
    latent, cond = generate_latent_points(n_batch)
    # predict outputs
    generated = generator.predict([latent, cond])
    return [generated, cond]


def generate(cond):
    latent = np.random.normal(size=(1, latent_dim))
    cond = np.expand_dims(cond, 0)
    return generator.predict([latent, cond])


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


class RandomWeightedAverage(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        global batch_size
        alpha = tf.random.uniform((batch_size,1,  1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class GradientPenalty(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GradientPenalty, self).__init__(**kwargs)

    def build(self, input_shapes):
        # Create a trainable weight variable for this layer.
        super(GradientPenalty, self).build(input_shapes)  # Be sure to call this somewhere!

    def call(self, inputs):
        target, wrt = inputs
        grad = K.gradients(target, wrt)[0]
        #TODO: check this formula! this seems different from https://github.com/kongyanye/cwgan-gp/blob/master/cwgan_gp.py
        #return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))-1
        gradients_sqr = K.square(grad)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        #gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        #return K.mean(gradient_penalty)
        # since we do it with a layer instead a loss function,
        #we dont do the final calculation here, but with the MSE loss function
        return 1 - gradient_l2_norm

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


# pixel-wise feature vector normalization layer
# from https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/
class PixelNormalization(tf.keras.layers.Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate square pixel values
        values = inputs ** 2.0
        # calculate the mean pixel values
        mean_values = K.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = K.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape


def create_generator():
    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(PixelNormalization())
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(PixelNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(1, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    # model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(n_classes, latent_dim)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)

def create_discriminator():
    model = Sequential()

    model.add(Dense(7 * 7 * 128, input_dim=np.prod(img_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2D(16, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    # model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(n_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])

    validity = model(model_input)

    return Model([img, label], validity)

print('building networks')
generator = create_generator()
critic = create_discriminator()
generator.trainable = False
# Image input (real sample)
real_img = tf.keras.layers.Input(shape=img_shape)
# Noise input
z_disc = tf.keras.layers.Input(shape=(latent_dim,))
# Generate image based of noise (fake sample) and add label to the input
label = tf.keras.layers.Input(shape=(1,))
fake_img = generator([z_disc, label])
# Discriminator determines validity of the real and fake images
fake = critic([fake_img, label])
valid = critic([real_img, label])

# Construct weighted average between real and fake images
interpolated_img = RandomWeightedAverage()([real_img, fake_img])

# Determine validity of weighted sample
validity_interpolated = critic([interpolated_img, label])
# here we use the approach from https://github.com/jleinonen/geogan/blob/master/geogan/gan.py,
# where gradient panely is a keras layer, and then 'mse' used as loss for this output
disc_gp = GradientPenalty()([validity_interpolated, interpolated_img])

# default from https://arxiv.org/pdf/1704.00028.pdf
#the choice of optimizer and its params might be very important
# i did some experimentation, and it seemst that Adam(lr=0.0001, beta_1=0.5, beta_2=0.9)
# is kind of unstable, the loss functions get bery high aboslute values, but
# in the end it manages to stabilize (even though at high values)
# this does not seem to happen  with Adam(lr=0.0001, beta_1=0, beta_2=0.9)
#  however, I am not sure whether this is not a problem of the implmenentaion here
#
# optimizer = tf.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.9)
optimizer = tf.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.9)
# with adam, the "runaway competition" ptoblem occurs sometimes (not always)
#optimizer = tf.optimizers.RMSprop(lr=0.00005)

critic_model = tf.keras.Model(inputs=[real_img, label, z_disc], outputs=[valid, fake, disc_gp])
critic_model.compile(loss=[wasserstein_loss,
                                wasserstein_loss,
                                'mse'],
                          optimizer=optimizer,
                          loss_weights=[1, 1, 10])

# For the generator we freeze the critic's layers
critic.trainable = False
generator.trainable = True

# Sampled noise for input to generator
z_gen = Input(shape=(latent_dim,))
# add label to the input
label = tf.keras.layers.Input(shape=(1,))
# Generate images based of noise
img = generator([z_gen, label])
# Discriminator determines validity
valid = critic([img, label])
# Defines generator model
generator_model = tf.keras.Model([z_gen, label], valid)
generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)
print('finished building networks')



hist = {'d_loss': [], 'g_loss': []}
print(f'start training on {n_samples} samples')


n_epochs=20
# create a dataqueue with the keras facilities. this allows
# to prepare the data in parallel to the training
sample_dataqueue = GeneratorEnqueuer(generate_real_samples(batch_size),
                                     use_multiprocessing=True)
sample_dataqueue.start(workers=2, max_queue_size=10)
sample_gen = sample_dataqueue.get()

# targets for loss function
gan_sample_dataqueue = GeneratorEnqueuer(generate_latent_points_as_generator(batch_size),
                                     use_multiprocessing=True)
gan_sample_dataqueue.start(workers=2, max_queue_size=10)
gan_sample_gen = gan_sample_dataqueue.get()

# targets for loss function
valid = -np.ones((batch_size, 1))
fake = np.ones((batch_size, 1))
dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty

bat_per_epo = int(n_samples / batch_size)

# we need to call the discriminator once in order
# to initialize the input shapes
[X_real, cond_real] = next(sample_gen)
latent = np.random.normal(size=(batch_size, latent_dim))
critic_model.predict([X_real, cond_real, latent])


start_epoch=0
for i in trange(n_epochs):
    epoch = 1 + i + start_epoch
    # enumerate batches over the training set
    for j in trange(bat_per_epo):

        for _ in range(n_disc):
            # fetch a batch from the queue
            [X_real, cond_real] = next(sample_gen)
            latent = np.random.normal(size=(batch_size, latent_dim))
            d_loss = critic_model.train_on_batch([X_real, cond_real,latent], [valid, fake, dummy])
            # we get for losses back here. average, valid, fake, and gradient_penalty
            # we want the average
            d_loss = d_loss[0]


        # train generator
        # prepare points in latent space as input for the generator
        [latent, cond] = next(gan_sample_gen)
        # update the generator via the discriminator's error
        g_loss = generator_model.train_on_batch([latent, cond], valid)
        # summarize loss on this batch
        print(f'{epoch}, {j + 1}/{bat_per_epo}, d_loss {d_loss}' + \
              f' g:{g_loss} ')  # , d_fake:{d_loss_fake} d_real:{d_loss_real}')

        if np.isnan(g_loss) or np.isnan(d_loss):
            raise ValueError('encountered nan in g_loss and/or d_loss')

        hist['d_loss'].append(d_loss)
        hist['g_loss'].append(g_loss)


    # plot generated examples
    fig1 = plt.figure(figsize=(25, 25))
    n_plot = 16
    X_fake, cond_fake = generate_fake_samples(n_plot)
    for iplot in range(n_plot):
        plt.subplot(4, 4, iplot +1)
        plt.imshow(X_fake[iplot].squeeze(), cmap='gray')
        plt.axis('off')
    plt.colorbar()
    plt.suptitle(f'epoch {epoch:04d}')
    plt.savefig(f'{plotdir}/fake_samples__{epoch:04d}_{j:06d}.{plot_format}')

    # plot loss
    fig2 = plt.figure()
    plt.plot(hist['d_loss'], label='d_loss')
    plt.plot(hist['g_loss'], label='g_loss')
    plt.ylabel('batch')
    plt.legend()
    plt.savefig(f'{plotdir}/training_loss_.{plot_format}')
    pd.DataFrame(hist).to_csv('hist.csv')
    display.display(fig1,fig2)

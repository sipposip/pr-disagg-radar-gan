#! /pfs/nobackup/home/s/sebsc/miniconda3/envs/pr-disagg-env/bin/python
#SBATCH -A SNIC2019-3-611
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:v100:1
"""
training script for the network. it loads the data as memmap, so it does not need large amounts of RAM

input: output from reformat_data.py and compute_valid_indices.py


@internal: run on kebnekaise (using sbatch definitions on top of the file) and on colab. final run
make on kebnekaise

on colab add the following on top of the first cell:
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
from google.colab import drive
drive.mount('/content/drive')


terminology used here: the word "generator" is used both for the generator of the GAN, and for
"python generators", which is a special type of iterable in python that we use here for feeding
the input data into the network.

@author: Sebastian Scher

needs tensorflow >=2.1
conda install tensorflow-gpu==2.1.0

"""
import pickle
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib

matplotlib.use('agg')
from pylab import plt
from tqdm import trange
from skimage.util import view_as_windows
from matplotlib.colors import LogNorm
from tensorflow.keras.utils import GeneratorEnqueuer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input


startdate = '20090101'
enddate = '20161231'

ndomain = 16  # gridpoints
stride = 16
tres = 1

tp_thresh_daily = 5  # mm. in the radardate the unit is mm/h, but then on 5 minutes steps.
# the conversion is done automatically in this script
n_thresh = 20

# normalization of daily sums
# we ues the 99.9 percentile of 2010
norm_scale = 127.4

# neural network parameters
n_disc = 5
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
latent_dim = 100
batch_size = 32 # this is used as global variable in randomweightedaverage
# the training is done with increasing batch size. each tuple is
# a combination nof number of epochs and batch_size
#n_epoch_and_batch_size_list = ((5, 32), (10, 64), (10, 128), (20, 256))
n_epoch_and_batch_size_list = ((50, 32),)

plot_format = 'png'

name='wgancp_pixelnorm_lon'

# input and output directories. different for different machines
if 'SNIC_RESOURCE' in os.environ.keys() and os.environ['SNIC_RESOURCE'] == 'kebnekaise':
    machine = 'kebnekaise'
else:
    machine = 'colab'

plotdirs ={'kebnekaise': f'plots_{name}/',
           'misu160': f'plots_{name}/',
           'colab':f'/content/drive/My Drive/data/smhi_radar/plots_{name}/'}
plotdir = plotdirs[machine]

outdirs = {'kebnekaise': f'/pfs/nobackup/home/s/sebsc/pr_disagg/trained_models/{name}/',
           'misu160': f'/climstorage/sebastian/pr_disagg/smhi/rained_models/{name}/',
           'colab': f'/content/drive/My Drive/data/smhi_radar/trained_models/{name}/'}
outdir = outdirs[machine]
# note for colab: sometimes mkdir does not work that way. in this case
# you have to create the directories manually
os.system(f'mkdir -p {plotdir}')
os.system(f'mkdir -p {outdir}')

# load data and precomputed indices

converted_data_paths = {'misu160': '/climstorage/sebastian/pr_disagg/smhi/preprocessed/',
                        'kebnekaise': '/home/s/sebsc/pfs/pr_disagg/smhi_radar/preprocessed',
                        'colab': '/content/drive/My Drive/data/smhi_radar/preprocessed/'}
converted_data_path = converted_data_paths[machine]
indices_data_paths = {'misu160': 'data/',
                      'kebnekaise': 'data/',
                      'colab': '/content/drive/My Drive/data/smhi_radar/preprocessed/'}
indices_data_path = indices_data_paths[machine]

data_ifile = f'{converted_data_path}/{startdate}-{enddate}_tres{tres}.npy'

params = f'{startdate}-{enddate}-tp_thresh_daily{tp_thresh_daily}_n_thresh{n_thresh}_ndomain{ndomain}_stride{stride}'
indices_file = f'{indices_data_path}/valid_indices_smhi_radar_{params}.pkl'

print('loading data')
# load the data as memmap
data = np.load(data_ifile, mmap_mode='r')

indices_all = pickle.load(open(indices_file, 'rb'))
# convert to array
indices_all = np.array(indices_all)
# this has shape (nsamples,3)
# each row is (tidx,yidx,xidx)

# for lon as additional input we need the max and min lonindex (for normalization)
min_lonidx = np.min(indices_all[:,2])
max_lonidx = np.max(indices_all[:,2])



print('finished loading data')

# the data has dimensions (sample,hourofday,x,y)
n_days, nhours, ny, nx = data.shape
n_channel=2 # precipitation plus lon
# sanity checks
assert (len(data.shape) == 4)
assert (len(indices_all.shape) == 2)
assert (indices_all.shape[1] == 3)
assert (nhours == 24 // tres)
assert (np.max(indices_all[:, 0]) < n_days)
assert (np.max(indices_all[:, 1]) < ny)
assert (np.max(indices_all[:, 2]) < nx)
assert (data.dtype == 'float32')

n_samples = len(indices_all)


def generate_real_samples(n_batch):
    """get random sampples and do the last preprocessing on them"""
    while True:
        # get random sample of indices from the precomputed indices
        # for this we generate random indices for the index list (confusing termoonology, since we use
        # indices to index the list of indices...
        ixs = np.random.randint(n_samples, size=n_batch)
        idcs_batch = indices_all[ixs]

        # now we select the data corresponding to these indices

        data_wview = view_as_windows(data, (1, 1, ndomain, ndomain))[..., 0, 0, :,:]
        batch = data_wview[idcs_batch[:, 0], :, idcs_batch[:, 1], idcs_batch[:, 2]]
        # add empty channel dimension (necessary for keras, which expects a channel dimension)
        batch = np.expand_dims(batch, -1)
        # compute daily sum (which is the condition)
        batch_cond = np.sum(batch, axis=1) # daily sum

        # the data now is in mm/hour, but we want it as fractions of the daily sum for each day
        for i in range(n_batch):
            batch[i] = batch[i] / batch_cond[i]

        # normalize daily sum
        batch_cond = batch_cond / norm_scale

        # add lon
        batch_lon = idcs_batch[:,2]
        # normalize to [0,1]
        batch_lon = (batch_lon - min_lonidx) / max_lonidx
        # repeat it to ndomain x ndomain x 1  (last is channel dimension)
        batch_lon = np.tile(batch_lon,(1,ndomain,ndomain,1)).T
        # now it has n_batch x ndomain x ndomain x 1

        # and add it as additional varialbes
        batch_cond = np.concatenate([batch_cond,batch_lon], axis=-1)

        assert (batch.shape == (n_batch, nhours, ndomain, ndomain, 1))
        assert (batch_cond.shape == (n_batch, ndomain, ndomain, n_channel))
        assert (~np.any(np.isnan(batch)))
        assert (~np.any(np.isnan(batch_cond)))
        assert (np.max(batch) <= 1)
        assert (np.min(batch) >= 0)

        yield [batch, batch_cond]


def generate_latent_points(n_batch):
    # generate points in the latent space and a random condition
    latent = np.random.normal(size=(n_batch, latent_dim))
    # randomly select conditions
    ixs = np.random.randint(0, n_samples, size=n_batch)
    idcs_batch = indices_all[ixs]

    data_wview = view_as_windows(data, (1, 1, ndomain, ndomain))[..., 0, 0, :,:]
    batch = data_wview[idcs_batch[:, 0], :, idcs_batch[:, 1], idcs_batch[:, 2]]
    # add empty channel dimension (necessary for keras, which expects a channel dimension)
    batch = np.expand_dims(batch, -1)
    batch_cond = np.sum(batch, axis=1) # daily sum
    # normalize daily sum
    batch_cond = batch_cond / norm_scale
    # add lon
    batch_lon = idcs_batch[:, 2]
    # normalize to [0,1]
    batch_lon = (batch_lon - min_lonidx) / max_lonidx
    # repeat it to ndomain x ndomain x 1  (last is channel dimension)
    batch_lon = np.tile(batch_lon, (1, ndomain, ndomain, 1)).T
    # now it has n_batch x ndomain x ndomain x 1

    # and add it as additional varialbes
    batch_cond = np.concatenate([batch_cond, batch_lon], axis=-1)
    assert (batch_cond.shape == (n_batch, ndomain, ndomain, n_channel))
    assert (~np.any(np.isnan(batch_cond)))
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
        alpha = tf.random.uniform((batch_size,1, 1, 1, 1))
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
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))-1

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

def create_discriminator():
    # we add the condition as additional channel. For this we
    # expand its dimensions alon the nhours axis via linear scaling
    in_cond = tf.keras.layers.Input(shape=(ndomain, ndomain, n_channel))
    # add nhours dimension (size 1 for now)
    cond_expanded = tf.keras.layers.Reshape((1, ndomain, ndomain, n_channel))(in_cond)
    cond_expanded = tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, rep=nhours, axis=1))(
        cond_expanded)
    in_sample = tf.keras.layers.Input(shape=(nhours, ndomain, ndomain, 1))

    in_combined = tf.keras.layers.Concatenate(axis=-1)([in_sample, cond_expanded])
    kernel_size = (3, 3, 3)
    main_net = tf.keras.Sequential([

        tf.keras.layers.Conv3D(64, kernel_size=kernel_size, strides=2, input_shape=(nhours, ndomain, ndomain, n_channel+1),
                               padding="valid"),  # 11x7x7x32
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv3D(128, kernel_size=kernel_size, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv3D(256, kernel_size=kernel_size, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv3D(256, kernel_size=kernel_size, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='linear'),
    ])
    out = main_net(in_combined)
    model = tf.keras.Model(inputs=[in_sample, in_cond], outputs=out)

    return model


def create_generator():

    # for the moment, the flat approach is used
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    # define model

    n_nodes = 256 * 2 * 2 * 3
    in_latent = tf.keras.layers.Input(shape=(latent_dim,))
    # the condition is a 2d array (ndomain x ndomain), we simply flatten it
    in_cond = tf.keras.layers.Input(shape=(ndomain, ndomain, n_channel))
    in_cond_flat = tf.keras.layers.Flatten()(in_cond)
    in_combined = tf.keras.layers.Concatenate()([in_latent, in_cond_flat])

    main_net = tf.keras.Sequential([
        tf.keras.layers.Dense(n_nodes, kernel_initializer=init),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Reshape((3, 2, 2, 256)),

        tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
        tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same', kernel_initializer=init),
        PixelNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.2),

        tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
        tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same', kernel_initializer=init),
        PixelNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.2),

        tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=init),
        PixelNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        # output 24x16x16x1
        tf.keras.layers.Conv3D(1, (3, 3, 3), activation='linear', padding='same', kernel_initializer=init),
        # softmax per gridpoint, thus over nhours, which is axis 1 (Softmax also counts the batch axis)
        tf.keras.layers.Softmax(axis=1),
        # check for Nans (only for debugging)
        tf.keras.layers.Lambda(
            lambda x: tf.debugging.check_numerics(x, 'found nan in output of per_gridpoint_softmax')),

    ])

    out = main_net(in_combined)
    model = tf.keras.Model(inputs=[in_latent, in_cond], outputs=out)

    return model


print('building networks')
generator = create_generator()
critic = create_discriminator()
generator.trainable = False
# Image input (real sample)
real_img = tf.keras.layers.Input(shape=(nhours,ndomain,ndomain,1))
# Noise input
z_disc = tf.keras.layers.Input(shape=(latent_dim,))
# Generate image based of noise (fake sample) and add label to the input
label = tf.keras.layers.Input(shape=(ndomain, ndomain, n_channel))
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
optimizer = tf.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.9)

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
label = tf.keras.layers.Input(shape=(ndomain, ndomain, n_channel))
# Generate images based of noise
img = generator([z_gen, label])
# Discriminator determines validity
valid = critic([img, label])
# Defines generator model
generator_model = tf.keras.Model([z_gen, label], valid)
generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)
print('finished building networks')

# plot some real samples
# plot a couple of samples
plt.figure(figsize=(25, 25))
n_plot = 30
[X_real, cond_real] = next(generate_real_samples(n_plot))
for i in range(n_plot):
    plt.subplot(n_plot, 25, i * 25 + 1)
    plt.imshow(cond_real[i, :, :,0], cmap=plt.cm.gist_earth_r, norm=LogNorm(vmin=0.01, vmax=1))
    plt.axis('off')
    for j in range(1, 24):
        plt.subplot(n_plot, 25, i * 25 + j + 1)
        plt.imshow(X_real[i, j, :, :].squeeze(), vmin=0, vmax=1, cmap=plt.cm.hot_r)
        plt.axis('off')
plt.colorbar()
plt.savefig(f'{plotdir}/real_samples.{plot_format}')

hist = {'d_loss': [], 'g_loss': []}
print(f'start training on {n_samples} samples')


def train(n_epochs, _batch_size, start_epoch=0):
    """
        train with fixed batch_size for given epochs
        make some example plots and save model after each epoch
    """
    global batch_size
    batch_size = _batch_size
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
                # we want the average of valid and fake
                d_loss = np.mean([d_loss[1], d_loss[2]])


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
        plt.figure(figsize=(25, 25))
        n_plot = 30
        X_fake, cond_fake = generate_fake_samples(n_plot)
        for iplot in range(n_plot):
            plt.subplot(n_plot, 25, iplot * 25 + 1)
            plt.imshow(cond_fake[iplot, :, :,0], cmap=plt.cm.gist_earth_r, norm=LogNorm(vmin=0.01, vmax=1))
            plt.axis('off')
            for jplot in range(1, 24):
                plt.subplot(n_plot, 25, iplot * 25 + jplot + 1)
                plt.imshow(X_fake[iplot, jplot, :, :,0], vmin=0, vmax=1, cmap=plt.cm.hot_r)
                plt.axis('off')
        plt.colorbar()
        plt.suptitle(f'epoch {epoch:04d}')
        plt.savefig(f'{plotdir}/fake_samples_{params}_{epoch:04d}_{j:06d}.{plot_format}')

        # plot loss
        plt.figure()
        plt.plot(hist['d_loss'], label='d_loss')
        plt.plot(hist['g_loss'], label='g_loss')
        plt.ylabel('batch')
        plt.legend()
        plt.savefig(f'{plotdir}/training_loss_{params}.{plot_format}')
        pd.DataFrame(hist).to_csv('hist.csv')
        plt.close('all')

        generator.save(f'{outdir}/gen_{params}_{epoch:04d}.h5')
        critic.save(f'{outdir}/disc_{params}_{epoch:04d}.h5')


# the training is done with increasing batch size,
# as defined in n_epoch_and_batch_size_list at the beginning of the script
start_epoch = 0
for n_epochs, batch_size in  n_epoch_and_batch_size_list:
    train(n_epochs, batch_size, start_epoch)
    start_epoch = start_epoch + n_epochs #this is only needed for correct plot labelling


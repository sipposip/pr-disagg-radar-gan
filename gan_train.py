#! /pfs/nobackup/home/s/sebsc/miniconda3/envs/pr-disagg-env/bin/python


# SBATCH -A SNIC2019-3-611
# SBATCH --time=24:00:00
# SBATCH --gres=gpu:v100:1
"""
on misu160: #! /climstorage/sebastian/anaconda3/envs/pr-disagg-env/bin/python

"""
import pickle
import os
import queue
import threading
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

# session_conf = tf.compat.v1.ConfigProto(
#       intra_op_parallelism_threads=10,
#       inter_op_parallelism_threads=2)
# sess = tf.compat.v1.Session(config=session_conf)

startdate = '20090101'
enddate = '20091231'
# enddate='20171231'
ndomain = 16  # gridpoints
tres = 1

tp_thresh_daily = 5  # mm. in the radardate the unit is mm/h, but then on 5 minutes steps.
# the conversion is done automatically in this script
n_thresh = 20

n_epochs = 1000
batch_size = 32
clip_value = 0.01
n_disc = 5

if 'SNIC_RESOURCE' in os.environ.keys() and os.environ['SNIC_RESOURCE'] == 'kebnekaise':
    machine = 'kebnekaise'
else:
    machine = 'colab'

plotdir = 'plots_main/'
outdirs = {'kebnekaise': '/pfs/nobackup/home/s/sebsc/pr_disagg/trained_models/',
           'misu160': '/climstorage/sebastian/pr_disagg/smhi/rained_models/',
           'colab': '/content/drive/My Drive/data/smhi_radar/trained_models/'}
outdir = outdirs[machine]
os.system(f'mkdir -p {plotdir}')
os.system(f'mkdir -p {outdir}')

# load data and precomputed indices
print('loading data')
converted_data_paths = {'misu160': '/climstorage/sebastian/pr_disagg/smhi/preprocessed/',
                        'kebnekaise': '/home/s/sebsc/pfs/pr_disagg/smhi_radar/preprocessed',
                        'colab': '/content/drive/My Drive/data/smhi_radar/preprocessed/'}
converted_data_path = converted_data_paths[machine]
indices_data_paths = {'misu160': 'data/',
                      'kebnekaise': 'data/',
                      'colab': '/content/drive/My Drive/data/smhi_radar/preprocessed/'}
indices_data_path = indices_data_paths[machine]

data_ifile = f'{converted_data_path}/{startdate}-{enddate}_tres{tres}.np.npy'

params = f'{startdate}-{enddate}-tp_thresh_daily{tp_thresh_daily}_n_thresh{n_thresh}_ndomain{ndomain}'
indices_file = f'{indices_data_path}/valid_indices_smhi_radar_{params}.pkl'

data = np.load(data_ifile)

# add empty channel dmiensions
data = np.expand_dims(data, -1)

indices_all = pickle.load(open(indices_file, 'rb'))
# convert to array
indices_all = np.array(indices_all)
# this has shapes (nsamples,3)
# each ros is (tidx,yidx,xidx)
print('finished loading data')

# the data has dimensions (sample,hourofday,x,y)
n_days, nhours, ny, nx, n_channel = data.shape
# sanity checks
assert (len(data.shape) == 5)
assert (len(indices_all.shape) == 2)
assert (indices_all.shape[1] == 3)
assert (nhours == 24 // tres)
assert (np.max(indices_all[:, 0]) < n_days)
assert (np.max(indices_all[:, 1]) < ny)
assert (np.max(indices_all[:, 2]) < nx)
assert (data.dtype == 'float32')

n_samples = len(indices_all)

# compute daily sum as condition
dsum = data.sum(axis=1)

# normalization
norm_scale = np.nanmax(dsum)

dsum = dsum / norm_scale

# convert the subdaily data to fractions of the daily sum
# TODO: make this more efficient. 1) directly modify data without copying firs, 2) maybe ommit forloop
fractions = data.copy()
for i in range(n_days):
    fractions[i] = data[i] / data[i].sum(axis=0)  # sum over day

data = fractions

assert (np.nanmax(fractions) <= 1)
assert (np.nanmin(fractions) >= 0)

latent_dim = 1024

n_batch = 32


def generate_real_samples(n_batch):
    while True:
        # get random sample of indices from the precomputed indices
        # for this we generate random indices for the index list (confusing termoonology, since we use
        # indices to index the list of indices...
        ixs = np.random.randint(n_samples, size=n_batch)
        idcs_batch = indices_all[ixs]

        # now we select the data corresponding to these indices

        # # slow implementatnion. kept here as reference because
        # # it is easier to understand than the fast implementation
        # batch = np.empty((n_batch, nhours, ndomain, ndomain, n_channel), dtype='float32')
        # batch_cond = np.empty((n_batch, ndomain, ndomain, n_channel), dtype='float32')
        # for i in range(n_batch):
        #     tidx, iy,ix = idcs_batch[i]
        #     batch[i,:,:,:] = fractions[tidx, :, iy:iy+ndomain, ix:ix+ndomain]
        #     batch_cond[i,:,:] = dsum[tidx, iy:iy+ndomain, ix:ix+ndomain]

        # fast implementation (3 timeas as gast for n_batch 1024)
        data_wview = view_as_windows(data, (1, 1, ndomain, ndomain, 1))[..., 0, 0, 0, :, :, :]
        dsum_wview = view_as_windows(dsum, (1, ndomain, ndomain, 1))[..., 0, 0, :, :, :]
        batch = data_wview[idcs_batch[:, 0], :, idcs_batch[:, 1], idcs_batch[:, 2]]
        batch_cond = dsum_wview[idcs_batch[:, 0], idcs_batch[:, 1], idcs_batch[:, 2]]

        assert (batch.shape == (n_batch, nhours, ndomain, ndomain, 1))
        assert (batch_cond.shape == (n_batch, ndomain, ndomain, 1))
        assert (~np.any(np.isnan(batch)))
        assert (~np.any(np.isnan(batch_cond)))
        assert (np.max(batch) <= 1)
        assert (np.min(batch) >= 0)

        yield [batch, batch_cond]


def generate_latent_points(n_batch):
    # generate points in the latent space
    latent = np.random.normal(size=(n_batch, latent_dim))
    # randomly select conditions
    ixs = np.random.randint(0, n_samples, size=n_batch)
    idcs_batch = indices_all[ixs]
    # slow reference implementation
    # batch_cond = np.empty((n_batch, ndomain, ndomain, n_channel), dtype='float32')
    # for i in range(n_batch):
    #     tidx, iy,ix = idcs_batch[i]
    #     batch_cond[i,:,:] = dsum[tidx, iy:iy+ndomain, ix:ix+ndomain]

    dsum_wview = view_as_windows(dsum, (1, ndomain, ndomain, 1))[..., 0, 0, :, :, :]
    batch_cond = dsum_wview[idcs_batch[:, 0], idcs_batch[:, 1], idcs_batch[:, 2]]
    assert (batch_cond.shape == (n_batch, ndomain, ndomain, 1))
    assert (~np.any(np.isnan(batch_cond)))
    return [latent, batch_cond]


def generate_fake_samples(n_batch):
    # generate points in latent space
    latent, cond = generate_latent_points(n_batch)
    # predict outputs
    generated = gen.predict([latent, cond])
    return [generated, cond]


def generate(cond):
    latent = np.random.normal(size=(1, latent_dim))
    cond = np.expand_dims(cond, 0)
    return gen.predict([latent, cond])


def wasserstein_loss(y_true, y_pred):
    # we use -1 for fake, and +1 for real labels
    return tf.reduce_mean(y_true * y_pred)


# create a dataqueue with the keras facilities. this allows
# to prepare the data in parallel to the training


sample_dataqueue = GeneratorEnqueuer(generate_real_samples(batch_size),
                                     use_multiprocessing=False)
sample_dataqueue.start(workers=1, max_queue_size=10)
sample_gen = sample_dataqueue.get()

# optimizer recommended by WGAN paper
optimizer = tf.keras.optimizers.RMSprop(lr=0.00005)


def create_discriminator():
    # we add the condition as additional channel. For this we
    # expand its dimensions alon the nhours axis via linear scaling
    in_cond = tf.keras.layers.Input(shape=(ndomain, ndomain, 1))
    # add nhours dimension (size 1 for now)
    cond_expanded = tf.keras.layers.Reshape((1, ndomain, ndomain, 1))(in_cond)
    cond_expanded = tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, rep=nhours, axis=1))(
        cond_expanded)
    in_sample = tf.keras.layers.Input(shape=(nhours, ndomain, ndomain, 1))

    in_combined = tf.keras.layers.Concatenate(axis=-1)([in_sample, cond_expanded])
    kernel_size = (3, 3, 3)
    main_net = tf.keras.Sequential([

        tf.keras.layers.Conv3D(64, kernel_size=kernel_size, strides=2, input_shape=(nhours, ndomain, ndomain, 2),
                               padding="valid"),  # 11x7x7x32
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv3D(128, kernel_size=kernel_size, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.BatchNormalization(momentum=0.8),

        tf.keras.layers.Conv3D(256, kernel_size=kernel_size, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.BatchNormalization(momentum=0.8),

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
    # TODO: open questions: how to best include the condition
    # jussi has a flat input, so he flattens the condition,
    # and then adds this to the noise, so simply obtaining 1 flat input
    # weight initialization]
    # however, one could also argue that there is spatial structure in the condition,
    # and we should use convolution first. so do a architecture similar to the discrimnator
    # to do some convs and then a flattening, and use this as input
    # we could also check how this is done for image stuff (there probably is image stuff that uses images as
    # condition)
    # for the moment, the flat approach is used
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    # define model

    n_nodes = 256 * 2 * 2 * 3
    in_latent = tf.keras.layers.Input(shape=(latent_dim,))
    in_cond = tf.keras.layers.Input(shape=(ndomain, ndomain, n_channel))
    in_cond_flat = tf.keras.layers.Flatten()(in_cond)
    in_combined = tf.keras.layers.Concatenate()([in_latent, in_cond_flat])

    main_net = tf.keras.Sequential([
        tf.keras.layers.Dense(n_nodes, kernel_initializer=init),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Reshape((3, 2, 2, 256)),

        tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
        tf.keras.layers.Conv3D(256, (3, 3, 3), padding='same', kernel_initializer=init),
        tf.keras.layers.BatchNormalization(momentum=0.8),
        tf.keras.layers.LeakyReLU(alpha=0.2),

        tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
        tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same', kernel_initializer=init),
        tf.keras.layers.BatchNormalization(momentum=0.8),
        tf.keras.layers.LeakyReLU(alpha=0.2),

        tf.keras.layers.UpSampling3D(size=(2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=init),
        tf.keras.layers.BatchNormalization(momentum=0.8),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        # output 24x16x16x1
        tf.keras.layers.Conv3D(1, (3, 3, 3), activation='linear', padding='same', kernel_initializer=init),
        tf.keras.layers.Activation(lambda x: tf.keras.activations.softmax(x, axis=1)),
        # softmax per gridpoint, thus over nhours
        # tf.keras.layers.Activation('tanh'),
        # check for Nans (only for debugging)
        tf.keras.layers.Lambda(
            lambda x: tf.debugging.check_numerics(x, 'found nan in output of per_gridpoint_softmax')),

    ])

    out = main_net(in_combined)
    model = tf.keras.Model(inputs=[in_latent, in_cond], outputs=out)

    return model


disc = create_discriminator()
disc.summary()
gen = create_generator()
gen.summary()

disc = tf.keras.Model(inputs=disc.inputs, outputs=disc.outputs)
disc.compile(loss=wasserstein_loss, optimizer=optimizer)
gen = tf.keras.Model(inputs=gen.inputs, outputs=gen.outputs)
disc.trainable = False
discriminator_trainable_weights = len(disc.trainable_weights)  # for asserts, below
generator_trainable_weights = len(gen.trainable_weights)

gen_latent, gen_cond = gen.inputs
# get output from the generator model
gen_output = gen.output
# connect  output and cond input from generator as inputs to discriminator
gan_output = disc([gen_output, gen_cond])
# define gan model as taking noise and cond and outputting a classification
gan = tf.keras.Model([gen_latent, gen_cond], gan_output)
# compile model
gan.compile(loss=wasserstein_loss, optimizer=optimizer)

# plot some real samples
# plot a couple of samples
plt.figure(figsize=(25, 10))
n_plot = batch_size
[X_real, cond_real] = next(sample_gen)
for i in range(n_plot):
    plt.subplot(n_plot, 25, i * 25 + 1)
    plt.imshow(cond_real[i, :, :].squeeze(), cmap=plt.cm.gist_earth_r, norm=LogNorm(vmin=0.01, vmax=1))
    plt.axis('off')
    for j in range(1, 24):
        plt.subplot(n_plot, 25, i * 25 + j + 1)
        plt.imshow(X_real[i, j, :, :].squeeze(), vmin=0, vmax=1, cmap=plt.cm.hot_r)
        plt.axis('off')
plt.colorbar()
plt.savefig(f'{plotdir}/real_samples.svg')

valid = np.ones((batch_size, 1))
fake = -np.ones((batch_size, 1))

bat_per_epo = int(n_samples / batch_size)
hist = {'d_loss': [], 'g_loss': []}
print('start training')
# train the generator and discriminator
# manually enumerate epochs
for i in trange(n_epochs):
    # enumerate batches over the training set
    for j in trange(bat_per_epo):

        for _ in range(n_disc):
            # train discrmininator
            disc.trainable = True
            # # get randomly selected 'real' samples
            # [X_real, cond_real] = generate_real_samples(batch_size)
            # # generate 'fake' examples
            # [X_fake, cond_fake]= generate_fake_samples(batch_size)

            # fetch a batch from the queue
            [X_real, cond_real] = next(sample_gen)
            X_fake, cond_fake = generate_fake_samples(batch_size)
            d_loss_fake = disc.train_on_batch([X_fake, cond_fake], fake)
            d_loss_real = disc.train_on_batch([X_real, cond_real], valid)
            d_loss = np.mean([d_loss_real, d_loss_fake])

            # Clip discriminator weights
            for l in disc.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                l.set_weights(weights)

        # train generator
        disc.trainable = False
        # prepare points in latent space as input for the generator
        [latent, cond] = generate_latent_points(batch_size)
        # update the generator via the discriminator's error
        g_loss = gan.train_on_batch([latent, cond], valid)
        # summarize loss on this batch
        print(f'{i + 1}, {j + 1}/{bat_per_epo}, d_loss {d_loss}' + \
              f' g:{g_loss} ')  # , d_fake:{d_loss_fake} d_real:{d_loss_real}')

        if np.isnan(g_loss) or np.isnan(d_loss):
            raise ValueError('encountered nan in g_loss and/or d_loss')

        hist['d_loss'].append(d_loss)
        hist['g_loss'].append(g_loss)

        if j % 50 == 0:
            # plot generated examples
            plt.figure(figsize=(25, 10))
            n_plot = 10
            for iplot in range(n_plot):
                plt.subplot(n_plot, 25, iplot * 25 + 1)
                plt.imshow(cond_fake[iplot, :, :].squeeze(), cmap=plt.cm.gist_earth_r, norm=LogNorm(vmin=0.01, vmax=1))
                plt.axis('off')
                for jplot in range(1, 24):
                    plt.subplot(n_plot, 25, iplot * 25 + jplot + 1)
                    plt.imshow(X_fake[iplot, jplot, :, :].squeeze(), vmin=0, vmax=1, cmap=plt.cm.hot_r)
                    plt.axis('off')
            plt.colorbar()
            plt.savefig(f'{plotdir}/fake_samples{i:04d}_{j:06d}.svg')

            # plot loss
            plt.plot()
            plt.plot(hist['d_loss'], label='d_loss')
            plt.plot(hist['g_loss'], label='g_loss')
            plt.ylabel('batch')
            plt.savefig(f'{plotdir}/training_loss.svg')
            pd.DataFrame(hist).to_csv('hist.csv')
            plt.close('all')



        # save networks every 100th batch (they are quite large)
        if i % 100 == 0:
            gen.save(f'{outdir}/gen_{i:04d}_{j:06d}.h5')
            disc.save(f'{outdir}/disc_{i:04d}_{j:06d}.h5')



#! /pfs/nobackup/home/s/sebsc/miniconda3/envs/pr-disagg-env/bin/python
#SBATCH -A SNIC2019-3-611
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --exclusive
"""
this script uses the trained generator to create precipitation scenarios.
a number of daily sum conditions are sampled from the test-data,
and for each sub-daily scenarios are generated with the generator.
The results are shown in various plots

this is the version were we have 2 additional channels for doy. therefore, in the places
where we actually need only the rainfall part of the condition, we have to select it with
[:,:,0] for a single condition and with [:,:,:,0] for a batch

"""

import pickle
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use('agg')
from pylab import plt
import seaborn as sns
import scipy.stats
from tqdm import trange
from skimage.util import view_as_windows
from matplotlib.colors import LogNorm
from tensorflow.keras import backend as K

# for reproducability, we set a fixed seed to the random number generator
np.random.seed(354)

# we need to specify train start and enddate to get correct filenames
train_startdate = '20090101'
train_enddate = '20161231'

eval_startdate = '20170101'
eval_enddate = '20181231'

# parameters (need to be the same as in training)
ndomain = 16  # gridpoints
stride = 16
tres = 1
latent_dim = 100

tp_thresh_daily = 5  # mm. in the radardate the unit is mm/h, but then on 5 minutes steps.
# the conversion is done automatically in this script
n_thresh = 20

# here we need to choose which epoch we use from the saved models (we saved them at the end of every
# epoch). visual inspection of the images generated from the training set showed
# that after epoch 20, things starts to detoriate. Therefore we use epoch 20.
epoch = 20
# normalization of daily sums
# we ues the 99.9 percentile of 2010
norm_scale = 127.4

plot_format = 'png'

name = 'wgancp_pixelnorm_doy'

# input and output directories. different for different machines
machine = 'kebnekaise'


plotdirs = {'kebnekaise': f'plots_generated_{name}/',
            'misu160': f'plots_generated_{name}/',
            'colab': f'/content/drive/My Drive/data/smhi_radar/plots_generated_{name}/'}
plotdir = plotdirs[machine]

outdirs = {'kebnekaise': f'/pfs/nobackup/home/s/sebsc/pr_disagg/trained_models/{name}/',
           'misu160': f'/climstorage/sebastian/pr_disagg/smhi/rained_models/{name}/',
           'colab': f'/content/drive/My Drive/data/smhi_radar/trained_models/{name}/'}
outdir = outdirs[machine]
# note for colab: sometimes mkdir does not work that way. in this case
# you have to create the directories manually
os.system(f'mkdir -p {plotdir}')
os.system(f'mkdir -p {outdir}')

# load data and precomputed indices for the test data

converted_data_paths = {'misu160': '/climstorage/sebastian/pr_disagg/smhi/preprocessed/',
                        'kebnekaise': '/home/s/sebsc/pfs/pr_disagg/smhi_radar/preprocessed',
                        'colab': '/content/drive/My Drive/data/smhi_radar/preprocessed/'}
converted_data_path = converted_data_paths[machine]
indices_data_paths = {'misu160': 'data/',
                      'kebnekaise': 'data/',
                      'colab': '/content/drive/My Drive/data/smhi_radar/preprocessed/'}
indices_data_path = indices_data_paths[machine]

data_ifile = f'{converted_data_path}/{eval_startdate}-{eval_enddate}_tres{tres}.npy'

params = f'{train_startdate}-{train_enddate}-tp_thresh_daily{tp_thresh_daily}_n_thresh{n_thresh}_ndomain{ndomain}_stride{stride}'
params_eval = f'{eval_startdate}-{eval_enddate}-tp_thresh_daily{tp_thresh_daily}_n_thresh{n_thresh}_ndomain{ndomain}_stride{stride}'
indices_file = f'{indices_data_path}/valid_indices_smhi_radar_{params_eval}.pkl'
timelist_ifile=f'{converted_data_path}/{eval_startdate}-{eval_enddate}_tres{tres}_doy.npy'
print('loading data')
# load the data as memmap
data = np.load(data_ifile, mmap_mode='r')

indices_all = pickle.load(open(indices_file, 'rb'))
# convert to array
indices_all = np.array(indices_all)
# this has shape (nsamples,3)
# each row is (tidx,yidx,xidx)
print('finished loading data')

timelist_all = np.load(timelist_ifile)

# the data has dimensions (sample,hourofday,x,y)
n_days, nhours, ny, nx = data.shape
n_channel = 3
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

print(f'evaluate in {n_samples} samples')

print('load the trained generator')
generator_file = f'{outdir}/gen_{params}_{epoch:04d}.h5'

# we need the custom layer PixelNormalization to load the generator
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


gen = tf.keras.models.load_model(generator_file, compile=False,
                                 custom_objects={'PixelNormalization': PixelNormalization})


# in order to use the model, we need to compile it (even though we dont need the los function
# and optimizer here, since we only do prediction)
def wasserstein_loss(y_true, y_pred):
    # we use -1 for fake, and +1 for real labels
    return tf.reduce_mean(y_true * y_pred)

gen.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(lr=0.00005))


def generate_real_samples_and_conditions(n_batch):
    """get random sampples and do the last preprocessing on them"""
    # get random sample of indices from the precomputed indices
    # for this we generate random indices for the index list (confusing termoonology, since we use
    # indices to index the list of indices...
    ixs = np.random.randint(n_samples, size=n_batch)
    idcs_batch = indices_all[ixs]

    # now we select the data corresponding to these indices
    data_wview = view_as_windows(data, (1, 1, ndomain, ndomain))[..., 0, 0, :, :]
    batch = data_wview[idcs_batch[:, 0], :, idcs_batch[:, 1], idcs_batch[:, 2]]
    # add empty channel dimension (necessary for keras, which expects a channel dimension)
    batch = np.expand_dims(batch, -1)
    # compute daily sum (which is the condition)
    batch_cond = np.sum(batch, axis=1)  # daily sum

    # the data now is in mm/hour, but we want it as fractions of the daily sum for each day
    for i in range(n_batch):
        batch[i] = batch[i] / batch_cond[i]

    # normalize daily sum
    batch_cond = batch_cond / norm_scale

    # add doy
    batch_doy = timelist_all[idcs_batch[:, 0]]
    # repeat it to ndomain x ndomain x 1  (last is channel dimension)
    batch_doy = np.tile(batch_doy, (1, ndomain, ndomain, 1)).T
    # now it has n_batch x ndomain x ndomain x 1
    # compute cos and sin (this is necessary because doy is a circular variale)
    batch_doy_sin = np.sin(2 * np.pi * batch_doy / 365)
    batch_doy_cos = np.cos(2 * np.pi * batch_doy / 365)
    # and add it as additional varialbes
    batch_cond = np.concatenate([batch_cond, batch_doy_sin, batch_doy_cos], axis=-1)

    assert (batch.shape == (n_batch, nhours, ndomain, ndomain, 1))
    assert (batch_cond.shape == (n_batch, ndomain, ndomain, n_channel))
    assert (~np.any(np.isnan(batch)))
    assert (~np.any(np.isnan(batch_cond)))
    assert (np.max(batch) <= 1)
    assert (np.min(batch) >= 0)

    return [batch, batch_cond]


plt.rcParams['savefig.bbox'] = 'tight'
cmap = plt.cm.gist_earth_r
plotnorm = LogNorm(vmin=0.01, vmax=50)

# for each (real) condition, generate a couple of fake
# distributions, and plot them all together

n_to_generate = 20
n_per_batch = 10
n_batches = n_to_generate // n_per_batch
n_fake_per_real = 15
plotcount = 0
for ibatch in trange(n_batches):

    reals, conds = generate_real_samples_and_conditions(n_per_batch)

    for real, cond in zip(reals, conds):
        plotcount += 1
        # for each cond, make several predictions with different latent noise
        latent = np.random.normal(size=(n_fake_per_real, latent_dim))
        # for efficiency reason, we dont make a single forecast with the network, but
        # we batch all n_fake_per_real together
        cond_batch = np.repeat(cond[np.newaxis], repeats=n_fake_per_real, axis=0)
        generated = gen.predict([latent, cond_batch])

        # make a matrix of mapplots.
        # first column: condition (daily mean), the same for every row
        # first row: real fractions per hour
        # rest of the rows: generated fractions per hour, 1 row per realization
        fig = plt.figure(figsize=(25, 12))
        n_plot = n_fake_per_real + 1
        ax = plt.subplot(n_plot, 25, 1)
        # compute unnormalized daily sum. select dsum channel
        dsum = cond[:,:,0] * norm_scale
        plt.imshow(dsum, cmap=cmap, norm=plotnorm)
        plt.axis('off')
        ax.annotate('real', xy=(0, 0.5), xytext=(-5, 0), xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='right', va='center', rotation='vertical')
        ax.annotate(f'daily sum', xy=(0.5, 1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
        for jplot in range(1, 24 + 1):
            ax = plt.subplot(n_plot, 25, jplot + 1)
            plt.imshow(real[jplot - 1, :, :].squeeze(), vmin=0, vmax=1, cmap=plt.cm.hot_r)
            plt.axis('off')
            ax.annotate(f'{jplot:02d}'':00', xy=(0.5, 1), xytext=(0, 5),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
        # plot fake samples
        for iplot in range(n_fake_per_real):
            plt.subplot(n_plot, 25, (iplot + 1) * 25 + 1)
            plt.imshow(dsum, cmap=cmap, norm=plotnorm)
            plt.axis('off')
            for jplot in range(1, 24 + 1):
                plt.subplot(n_plot, 25, (iplot + 1) * 25 + jplot + 1)
                im = plt.imshow(generated[iplot, jplot - 1, :, :].squeeze(), vmin=0, vmax=1, cmap=plt.cm.hot_r)
                plt.axis('off')
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.007, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('fraction of daily precipitation', fontsize=16)
        cbar.ax.tick_params(labelsize=16)

        plt.savefig(f'{plotdir}/generated_fractions_{params}_{epoch:04d}_{plotcount:04d}.{plot_format}')

        # now the same, but showing absolute precipitation fields
        # compute absolute precipitation from fraction of daily sum.
        # this can be done with numpy broadcasting.
        # we also have to multiply with norm_scale (because cond is normalized)
        generated_scaled = generated.squeeze() * cond[:,:,0] * norm_scale

        real_scaled = real.squeeze() * cond[:,:,0] * norm_scale
        fig = plt.figure(figsize=(25, 12))
        # plot real one
        ax = plt.subplot(n_plot, 25, 1)
        im = plt.imshow(dsum, cmap=cmap, norm=plotnorm)
        plt.axis('off')
        ax.annotate('real', xy=(0, 0.5), xytext=(-5, 0), xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='right', va='center', rotation='vertical')
        ax.annotate(f'daily sum', xy=(0.5, 1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

        for jplot in range(1, 24 + 1):
            ax = plt.subplot(n_plot, 25, jplot + 1)
            plt.imshow(real_scaled[jplot - 1, :, :].squeeze(), cmap=cmap, norm=plotnorm)
            plt.axis('off')
            ax.annotate(f'{jplot:02d}'':00', xy=(0.5, 1), xytext=(0, 5),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
        # plot fake samples
        for iplot in range(n_fake_per_real):
            plt.subplot(n_plot, 25, (iplot + 1) * 25 + 1)
            plt.imshow(dsum, cmap=cmap, norm=plotnorm)
            plt.axis('off')
            for jplot in range(1, 24 + 1):
                plt.subplot(n_plot, 25, (iplot + 1) * 25 + jplot + 1)
                plt.imshow(generated_scaled[iplot, jplot - 1, :, :].squeeze(), cmap=cmap, norm=plotnorm)
                plt.axis('off')
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.007, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('precipitation [mm]', fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        plt.savefig(f'{plotdir}/generated_precip_{params}_{epoch:04d}_{plotcount:04d}.{plot_format}')

        plt.close('all')


# compute statistics over
# many generated smaples
# we compute the areamean,
n_sample = 10000
amean_fraction_gen = []
amean_fraction_real = []
amean_gen = []
amean_real = []
dists_real = []
dists_gen = []

# for each real conditoin, we crate 1 fake sample
for i in trange(n_sample):
    real, cond = generate_real_samples_and_conditions(1)
    latent = np.random.normal(size=(1, latent_dim))
    generated = gen.predict([latent, cond])

    generated = generated.squeeze()
    real = real.squeeze()
    cond = cond[:,:,0].squeeze()
    # compute area means
    amean_fraction_gen.append(np.mean(generated, axis=(1, 2)).squeeze())
    amean_fraction_real.append(np.mean(real, axis=(1, 2)).squeeze())
    amean_gen.append(np.mean(generated * cond * norm_scale, axis=(1, 2)).squeeze())
    amean_real.append(np.mean(real * cond * norm_scale, axis=(1, 2)).squeeze())
    dists_real.append(real * cond * norm_scale)
    dists_gen.append(generated * cond * norm_scale)


amean_fraction_gen = np.array(amean_fraction_gen)
amean_fraction_real = np.array(amean_fraction_real)
amean_gen = np.array(amean_gen)
amean_real = np.array(amean_real)
dists_gen = np.array(dists_gen)
dists_real = np.array(dists_real)


def ecdf(data):
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x, y)


sns.set_palette('colorblind')
# ecdf of area means. the hours are flattened
plt.figure()
ax1 = plt.subplot(211)
plt.plot(*ecdf(amean_gen.flatten()), label='gen')
plt.plot(*ecdf(amean_real.flatten()), label='real')
plt.legend(loc='upper left')
sns.despine()
plt.xlabel('mm/h')
plt.ylabel('ecdf areamean')
plt.semilogx()
# ecdf of (flattened) spatial data
ax2 = plt.subplot(212)
plt.plot(*ecdf(dists_gen.flatten()), label='gen')
plt.plot(*ecdf(dists_real.flatten()), label='real')
plt.legend(loc='upper left')
sns.despine()
plt.ylabel('ecdf')
plt.xlabel('mm/h')
plt.semilogx()
plt.tight_layout()
plt.savefig(f'{plotdir}/ecdf_allx_{params}_{epoch:04d}.png', dpi=400)
# cut at 0.1mm/h
ax1.set_xlim(xmin=0.5)
ax1.set_ylim(ymin=0.8, ymax=1.01)
ax2.set_xlim(xmin=0.1)
ax2.set_ylim(ymin=0.6, ymax=1.01)
plt.savefig(f'{plotdir}/ecdf_{params}_{epoch:04d}.png', dpi=400)

plt.close('all')
# free some memory
del dists_gen
del dists_real

# convert to pandas data frame, with timeofday ('hour') as additional column
res_df = []
for i in range(24):
    _df1 = pd.DataFrame({'fraction': amean_fraction_gen[:, i],
                         'precip': amean_gen[:, i],
                         'typ': 'generated',
                         'hour': i + 1}, index=np.arange(len(amean_gen)))
    _df2 = pd.DataFrame({'fraction': amean_fraction_real[:, i].squeeze(),
                         'precip': amean_real[:, i],
                         'typ': 'real',
                         'hour': i + 1}, index=np.arange(len(amean_gen)))
    res_df.append(_df1)
    res_df.append(_df2)


df = pd.concat(res_df)
df.to_csv(f'{plotdir}/gen_and_real_ameans_{params}_{epoch:04d}.csv')

# make boxplot
for showfliers in (True, False):

    plt.figure()
    plt.subplot(211)
    sns.boxplot('hour', 'precip', data=df, hue='typ', showfliers=showfliers)
    plt.xlabel('')
    sns.despine()
    plt.subplot(212)
    sns.boxplot('hour', 'fraction', data=df, hue='typ', showfliers=showfliers)
    sns.despine()
    plt.suptitle(f'n={n_sample}')
    plt.savefig(f'{plotdir}/daily_cycle_showfliers{showfliers}_{params}_{epoch:04d}.svg')


## for a single real one, generate a large
# number of fake distributions, and then
# plot the areamean in a lineplot
# we generate 100 fake distributions with different noise accross the samples
# and additionally 10 fake ones that use the same noise for all plots
# the latter we plot in the same color (1 seperate color for each generated one)
# so that we can compare them accross the plots

n_to_generate = 20
n_fake_per_real = 100
n_fake_per_real_samenoise = 10
plotcount = 0
hours = np.arange(1, 24 + 1)
# use same noise for all samples
latent_shared = np.random.normal(size=(n_fake_per_real_samenoise, latent_dim))
for isample in trange(n_to_generate):
    real, cond = generate_real_samples_and_conditions(1)
    latent= np.random.normal(size=(n_fake_per_real, latent_dim))
    # for efficiency reason, we dont make a single forecast with the network, but
    # we batch all n_fake_per_real together
    cond_batch = np.repeat(cond, repeats=n_fake_per_real, axis=0)
    cond_batch_samenoise = np.repeat(cond, repeats=n_fake_per_real_samenoise, axis=0)
    generated = gen.predict([latent, cond_batch], verbose=1)
    generated_samenoise = gen.predict([latent_shared, cond_batch_samenoise], verbose=1)
    real = real.squeeze()
    generated = generated.squeeze()
    generated_samenoise = generated_samenoise.squeeze()
    # compute area mean
    amean_real = np.mean(real * cond[:,:,0].squeeze() * norm_scale, (1, 2))
    amean_gen = np.mean(generated * cond[:,:,0].squeeze() * norm_scale, (2, 3))  # generated has a time dimension
    amean_gen_samenoise = np.mean(generated_samenoise * cond[:,:,0].squeeze() * norm_scale, (2, 3))  # generated has a time dimension

    plt.figure(figsize=(7, 3))
    plt.plot(hours, amean_gen.T, label='_nolegend_', alpha=0.3,color='#1b9e77')
    plt.plot(hours, amean_gen_samenoise.T, label='_nolegend_', alpha=1)
    plt.plot(hours, amean_real, label='real', color='black')
    plt.xlabel('hour')
    plt.ylabel('precipitation [mm/hour]')
    plt.legend()
    sns.despine()
    plt.savefig(f'{plotdir}/distribution_lineplot_samenosie_{params}_{epoch:04d}_{isample:04d}.svg')
    plt.close('all')

# take two conditions, and
# then plot the areamean of the resulting distributions, and check whether they are different
# we use the same noise for both, to avoid finding effects that only might come from the noise
n_fake_per_real = 1000
latent = np.random.normal(size=(n_fake_per_real, latent_dim))
for isample in trange(20):
    real1, cond1 = generate_real_samples_and_conditions(1)

    cond_batch1 = np.repeat(cond1, repeats=n_fake_per_real, axis=0)
    generated1 = gen.predict([latent, cond_batch1], verbose=1)
    real2, cond2 = generate_real_samples_and_conditions(1)
    cond_batch2 = np.repeat(cond2, repeats=n_fake_per_real, axis=0)
    generated2 = gen.predict([latent, cond_batch2], verbose=1)

    amean_fraction_real1 = np.mean(real1.squeeze(), (1, 2)).squeeze()
    amean_fraction_gen1 = np.mean(generated1, (2, 3)).squeeze()  # generated has a time dimension
    amean_fraction_real2 = np.mean(real2.squeeze(), (1, 2)).squeeze()
    amean_fraction_gen2 = np.mean(generated2.squeeze(), (2, 3)).squeeze()  # generated has a time dimension

    res_df = []
    for i in range(24):
        _df1 = pd.DataFrame({'fraction': amean_fraction_gen1[:, i],
                             'cond': 1,
                             'hour': i + 1}, index=np.arange(len(amean_fraction_gen1)))
        _df2 = pd.DataFrame({'fraction': amean_fraction_gen2[:, i],
                             'cond': 2,
                             'hour': i + 1}, index=np.arange(len(amean_fraction_gen1)))
        res_df.append(_df1)
        res_df.append(_df2)

    df = pd.concat(res_df)
    df.to_csv(f'{plotdir}/check_conditional_dist_samenoise_{params}_{epoch:04d}_{isample:04d}.csv')
    pvals_per_hour = []
    for hour in range(1,24+1):
        sub = df.query('hour==@hour')
        _, p = scipy.stats.ks_2samp(sub.query('cond==1')['fraction'], sub.query('cond==2')['fraction'])
        pvals_per_hour.append(p)
    np.savetxt(f'{plotdir}/check_conditional_dist_samenoise_KSpval{params}_{epoch:04d}_{isample:04d}.txt', pvals_per_hour)
    for showfliers in (True, False):
        fig = plt.figure(constrained_layout=True, figsize=(6, 4.8))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(cond1[:,:,0].squeeze(), cmap=cmap, norm=plotnorm)
        plt.title('cond 1')
        plt.axis('off')
        plt.colorbar(im)
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(cond2[:,:,0].squeeze(), cmap=cmap, norm=plotnorm)
        plt.title('cond 2')
        plt.axis('off')
        plt.colorbar(im)
        ax3 = fig.add_subplot(gs[1, :])
        sns.boxplot('hour', 'fraction', hue='cond', data=df, ax=ax3, showfliers=showfliers)
        sns.despine()
        plt.savefig(f'{plotdir}/check_conditional_dist_samenoise_showfliers{showfliers}_{params}_{epoch:04d}_{isample:04d}.svg')

    plt.close('all')

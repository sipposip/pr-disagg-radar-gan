
import pickle
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib

#matplotlib.use('agg')
from pylab import plt
from tqdm import trange
from skimage.util import view_as_windows
from matplotlib.colors import LogNorm
from tensorflow.keras.utils import GeneratorEnqueuer

train_startdate = '20100101'
train_enddate = '20101231'
eval_startdate = '20100101'
eval_enddate = '20101231'

ndomain = 16  # gridpoints
stride = 16
tres = 1

tp_thresh_daily = 5  # mm. in the radardate the unit is mm/h, but then on 5 minutes steps.
# the conversion is done automatically in this script
n_thresh = 20

epoch=14

# normalization of daily sums
# we ues the 99.9 percentile of 2010
norm_scale = 127.4

plot_format = 'png'

# input and output directories. different for different machines
if 'SNIC_RESOURCE' in os.environ.keys() and os.environ['SNIC_RESOURCE'] == 'kebnekaise':
    machine = 'kebnekaise'
else:
    machine = 'colab'

plotdirs ={'kebnekaise': 'plots_main/',
           'misu160': 'plots_main/',
           'colab':'/content/drive/My Drive/data/smhi_radar/plots_main/'}
plotdir = plotdirs[machine]

outdirs = {'kebnekaise': '/pfs/nobackup/home/s/sebsc/pr_disagg/trained_models/',
           'misu160': '/climstorage/sebastian/pr_disagg/smhi/rained_models/',
           'colab': '/content/drive/My Drive/data/smhi_radar/trained_models/'}
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

data_ifile = f'{converted_data_path}/{eval_startdate}-{eval_enddate}_tres{tres}.npy'

params = f'{train_startdate}-{train_enddate}-tp_thresh_daily{tp_thresh_daily}_n_thresh{n_thresh}_ndomain{ndomain}_stride{stride}'
indices_file = f'{indices_data_path}/valid_indices_smhi_radar_{params}.pkl'
print('loading data')
# load the data as memmap
data = np.load(data_ifile, mmap_mode='r')


indices_all = pickle.load(open(indices_file, 'rb'))
# convert to array
indices_all = np.array(indices_all)
# this has shape (nsamples,3)
# each row is (tidx,yidx,xidx)
print('finished loading data')

# the data has dimensions (sample,hourofday,x,y)
n_days, nhours, ny, nx = data.shape
n_channel=1
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

print('load the trained generator')
generator_file =f'{outdir}/gen_{params}_{epoch:04d}.h5'

gen = tf.keras.models.load_model(generator_file, compile=False)

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
    assert (batch.shape == (n_batch, nhours, ndomain, ndomain, 1))
    assert (batch_cond.shape == (n_batch, ndomain, ndomain, 1))
    assert (~np.any(np.isnan(batch)))
    assert (~np.any(np.isnan(batch_cond)))
    assert (np.max(batch) <= 1)
    assert (np.min(batch) >= 0)

    return [batch, batch_cond]




n_per_batch = 4
ibatch=0
n_fake_per_real = 30
latent_dim=1024
plotcount = 0
reals, conds = generate_real_samples_and_conditions(n_per_batch)


for real, cond in zip(reals,conds):
    plotcount+=1
    # for each cond, make several predictions with different latent noise
    latent = np.random.normal(size=(n_fake_per_real, latent_dim))
    # for efficiency reason, we dont make a single forecast with the network, but
    # we batch all n_fake_per_real together
    cond_batch = np.repeat(cond[np.newaxis],repeats=n_fake_per_real,axis=0)
    generated = gen.predict([latent, cond_batch])

    fig = plt.figure(figsize=(25, 25))
    n_plot = n_fake_per_real + 1
    # plot real one
    ax = plt.subplot(n_plot, 25, 1)
    plt.imshow(cond.squeeze(), cmap=plt.cm.gist_earth_r, norm=LogNorm(vmin=0.01, vmax=1))
    plt.ylabel('real')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    for jplot in range(1, 24):
        plt.subplot(n_plot, 25, jplot + 1)
        plt.imshow(real[jplot, :, :].squeeze(), vmin=0, vmax=1, cmap=plt.cm.hot_r)
        plt.axis('off')
    # plot fake samples
    for iplot in range(n_fake_per_real):
        plt.subplot(n_plot, 25, (iplot + 1) * 25 + 1)
        plt.imshow(cond.squeeze(), cmap=plt.cm.gist_earth_r, norm=LogNorm(vmin=0.01, vmax=1))
        plt.axis('off')
        for jplot in range(1, 24):
            plt.subplot(n_plot, 25, (iplot + 1) * 25 + jplot + 1)
            plt.imshow(generated[iplot, jplot, :, :].squeeze(), vmin=0, vmax=1, cmap=plt.cm.hot_r)
            plt.axis('off')
    fig.tight_layout()
    plt.colorbar()
    plt.savefig(f'{plotdir}/generated_fractions_{params}_{epoch:04d}_{plotcount:04d}.{plot_format}')


    # now the same, but showing absolute precipitation fields
    # compute absolute precipitation fram fraction of daily sum.
    # this can be done with numpy broadcasting
    generated_scaled = generated * cond

    real_scaled = real * cond
    fig = plt.figure(figsize=(25, 25))
    n_plot = n_fake_per_real + 1
    # plot real one
    ax = plt.subplot(n_plot, 25, 1)
    im = plt.imshow(cond.squeeze(), cmap=plt.cm.gist_earth_r, norm=LogNorm(vmin=0.001, vmax=0.2))
    plt.ylabel('real')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    for jplot in range(1, 24):
        plt.subplot(n_plot, 25, jplot + 1)
        plt.imshow(real_scaled[jplot, :, :].squeeze(), cmap=plt.cm.gist_earth_r, norm=LogNorm(vmin=0.001, vmax=0.2))
        plt.axis('off')
    # plot fake samples
    for iplot in range(n_fake_per_real):
        plt.subplot(n_plot, 25, (iplot + 1) * 25 + 1)
        plt.imshow(cond.squeeze(), cmap=plt.cm.gist_earth_r, norm=LogNorm(vmin=0.001, vmax=0.2))
        plt.axis('off')
        for jplot in range(1, 24):
            plt.subplot(n_plot, 25, (iplot + 1) * 25 + jplot + 1)
            plt.imshow(generated_scaled[iplot, jplot, :, :].squeeze(), cmap=plt.cm.gist_earth_r, norm=LogNorm(vmin=0.001, vmax=0.2))
            plt.axis('off')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f'{plotdir}/generated_precip_{params}_{epoch:04d}_{plotcount:04d}.{plot_format}')


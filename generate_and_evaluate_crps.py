#! /pfs/nobackup/home/s/sebsc/miniconda3/envs/pr-disagg-env/bin/python
#SBATCH -A SNIC2020-5-628
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:k80:1
"""
this script uses the trained generator to create precipitation scenarios.
a number of daily sum conditions are sampled from the test-data,
and for each sub-daily scenarios are generated with the generator.
The results are shown in various plots
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
import properscoring
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

name = 'wgancp_pixelnorm'

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
n_channel = 1
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
    assert (batch.shape == (n_batch, nhours, ndomain, ndomain, 1))
    assert (batch_cond.shape == (n_batch, ndomain, ndomain, 1))
    assert (~np.any(np.isnan(batch)))
    assert (~np.any(np.isnan(batch_cond)))
    assert (np.max(batch) <= 1)
    assert (np.min(batch) >= 0)

    return [batch, batch_cond]


plt.rcParams['savefig.bbox'] = 'tight'


# compute statistics over
# many generated smaples
# we compute the areamean,
n_sample = 1000
n_fake_per_real = 1000



baseline = np.load('rainfarm_calibration_data.npy')
dsum = np.sum(baseline, axis=1)  # daily sum
# the data now is in mm/hour, but we want it as fractions of the daily sum for each day
for i in range(len(baseline)):
    baseline[i] = baseline[i] / dsum[i]


# for each real conditoin, we crate fake_per_sample scenarios
crps_amean_all = []
crps_baseline_amean_all = []
for i in trange(n_sample):
    real, cond = generate_real_samples_and_conditions(1)
    cond_batch = np.repeat(cond, repeats=n_fake_per_real, axis=0)
    latent = np.random.normal(size=(n_fake_per_real, latent_dim))
    generated = gen.predict([latent, cond_batch])

    generated = generated.squeeze()
    real = real.squeeze()
    cond = cond.squeeze()
    real = (real * cond * norm_scale)
    generated = (generated * cond * norm_scale)
    crps = properscoring.crps_ensemble(real, generated, axis=0)
    # compute areamean crps
    crps_areamean = np.mean(crps,axis=(1,2))
    crps_amean_all.append(crps_areamean)


    baseline_scaled = baseline * cond * norm_scale
    crps_baseline = properscoring.crps_ensemble(real, baseline_scaled, axis=0)
    crps_baseline_amean = np.mean(crps_baseline, axis=(1,2))
    crps_baseline_amean_all.append(crps_baseline_amean)

crps_amean_all = np.array(crps_amean_all)
crps_baseline_amean_all = np.array(crps_baseline_amean_all)

crps_totalmean = np.mean(crps_amean_all)
crps_per_hour = np.mean(crps_amean_all, axis=0)
crps_baseline_totalmean = np.mean(crps_baseline_amean_all)
crps_baseline_per_hour = np.mean(crps_baseline_amean_all, axis=0)

pickle.dump((crps_amean_all, crps_baseline_amean_all), open('data/crps_results.pkl','wb'))



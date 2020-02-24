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
import pandas as pd
from tqdm import trange
from skimage.util import view_as_windows
from matplotlib.colors import LogNorm
from tensorflow.keras import backend as K

train_startdate = '20090101'
train_enddate = '20161231'
# eval_startdate = '20090101'
# eval_enddate = '20161231'

eval_startdate = '20170101'
eval_enddate = '20181231'

ndomain = 16  # gridpoints
stride = 16
tres = 1

tp_thresh_daily = 5  # mm. in the radardate the unit is mm/h, but then on 5 minutes steps.
# the conversion is done automatically in this script
n_thresh = 20

epoch = 50
#TODO HACK
epoch = 3
# normalization of daily sums
# we ues the 99.9 percentile of 2010
norm_scale = 127.4

plot_format = 'png'

name = 'wgancp_pixelnorm'

# input and output directories. different for different machines
if 'SNIC_RESOURCE' in os.environ.keys() and os.environ['SNIC_RESOURCE'] == 'kebnekaise':
    machine = 'kebnekaise'
else:
    machine = 'colab'

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
# colormap for precipitation, adapted from https://unidata.github.io/python-gallery/examples/Precipitation_Map.html
clevs = [0, 0.1, 0.3, 0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40,
         50, 70, 100, 150, 200, 250, 300, 400]
cmap_data = [(1.0, 1.0, 1.0),
             (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
             (0.0, 1.0, 1.0),
             (0.0, 0.8784313797950745, 0.501960813999176),
             (0.0, 0.7529411911964417, 0.0),
             (0.501960813999176, 0.8784313797950745, 0.0),
             (1.0, 1.0, 0.0),
             (1.0, 0.6274510025978088, 0.0),
             (1.0, 0.0, 0.0),
             (1.0, 0.125490203499794, 0.501960813999176),
             (0.9411764740943909, 0.250980406999588, 1.0),
             (0.501960813999176, 0.125490203499794, 1.0),
             (0.250980406999588, 0.250980406999588, 1.0),
             (0.125490203499794, 0.125490203499794, 0.501960813999176),
             (0.125490203499794, 0.125490203499794, 0.125490203499794),
             (0.501960813999176, 0.501960813999176, 0.501960813999176),
             (0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
             (0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
             (0.8549019694328308, 0.6509804129600525, 0.47058823704719543),
             (0.6274510025978088, 0.42352941632270813, 0.23529411852359772),
             (0.4000000059604645, 0.20000000298023224, 0.0)]
cmap = mcolors.ListedColormap(cmap_data, 'precipitation')
plotnorm = mcolors.BoundaryNorm(clevs, cmap.N)
cmap = plt.cm.gist_earth_r
plotnorm = LogNorm(vmin=0.01, vmax=50)

# for each (real) condition, generate a couple of fake
# distributions, and plot them all together

n_to_generate = 20
n_per_batch = 10
n_batches = n_to_generate // n_per_batch
n_fake_per_real = 15
latent_dim = 100
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
        # compute unnormalized daily sum. squeeze away empty channel dimension (for plotting)
        dsum = cond.squeeze() * norm_scale
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
        generated_scaled = generated * cond * norm_scale

        real_scaled = real * cond * norm_scale
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

#TODO: this could be made faster via batching the predictions (right now only single predictions)
n_sample = 5000
amean_fraction_gen = []
amean_fraction_real = []
amean_gen = []
amean_real = []

fraction_tmean_gen = np.zeros((nhours, ndomain, ndomain))
fraction_tmean_real = np.zeros((nhours, ndomain, ndomain))
tmean_gen = np.zeros((nhours, ndomain, ndomain))
tmean_real = np.zeros((nhours, ndomain, ndomain))

for i in trange(n_sample):
    real, cond = generate_real_samples_and_conditions(1)
    latent = np.random.normal(size=(1, latent_dim))
    generated = gen.predict([latent, cond])
    
    #NOTE: when batch_size >1, then the axis in the mean functions needs to be adapted!
    generated = generated.squeeze()
    real = real.squeeze()
    cond = cond.squeeze()
    # compute area means
    amean_fraction_gen.append(np.mean(generated, axis=(1, 2)).squeeze())
    amean_fraction_real.append(np.mean(real, axis=(1, 2)).squeeze())
    amean_gen.append(np.mean(generated*cond*norm_scale, axis=(1, 2)).squeeze())
    amean_real.append(np.mean(real*cond*norm_scale, axis=(1, 2)).squeeze())

    fraction_tmean_gen += generated.squeeze()/n_sample
    fraction_tmean_real += real.squeeze()/n_sample
    tmean_gen += (generated*cond*norm_scale).squeeze()/n_sample
    tmean_real += (real*cond*norm_scale).squeeze()/n_sample


amean_fraction_gen = np.array(amean_fraction_gen)
amean_fraction_real = np.array(amean_fraction_real)
amean_gen = np.array(amean_gen)
amean_real = np.array(amean_real)


# convert to pandas data frame, with time ofday ('hour') as additional column

res_df = []
for i in range(24):
    _df1 = pd.DataFrame({'fraction':amean_fraction_gen[:,i],
                         'precip': amean_gen[:,i],
                        'typ':'generated',
                       'hour':i+1}, index=np.arange(len(amean_gen)))
    _df2 = pd.DataFrame({'fraction': amean_fraction_real[:, i].squeeze(),
                         'precip': amean_real[:, i],
                        'typ': 'real',
                        'hour': i+1},  index=np.arange(len(amean_gen)))
    res_df.append(_df1)
    res_df.append(_df2)

df = pd.concat(res_df)
plt.figure()
plt.subplot(211)
sns.boxplot('hour', 'precip',data=df, hue='typ', showfliers=False)
plt.subplot(212)
sns.boxplot('hour', 'fraction',data=df, hue='typ', showfliers=False)
plt.suptitle(f'n={n_sample}')
plt.savefig(f'{plotdir}/daily_cycle_{params}_{epoch:04d}.svg')


fig = plt.figure(figsize=(25, 6))
for hour in range(24):

    ax = plt.subplot(4,24,hour+1)

    ax.annotate(f'{hour+1:02d}'':00', xy=(0.5, 1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
    if hour == 0:
        ax.annotate('frac real', xy=(0, 0.5), xytext=(-5, 0), xycoords='axes fraction', textcoords='offset points',
                     size='large', ha='right', va='center', rotation='vertical')
    im_frac = plt.imshow(fraction_tmean_real[hour], vmin=0, vmax=0.07, cmap=plt.cm.hot_r)
    plt.axis('off')
    if hour == 23:
        plt.colorbar()
    ax = plt.subplot(4, 24, hour + 1 + 24*1)
    if hour == 0:
        ax.annotate('frac gen', xy=(0, 0.5), xytext=(-5, 0), xycoords='axes fraction', textcoords='offset points',
                     size='large', ha='right', va='center', rotation='vertical')
    plt.imshow(fraction_tmean_gen[hour], vmin=0, vmax=0.07, cmap=plt.cm.hot_r)
    plt.axis('off')
    ax = plt.subplot(4, 24, hour + 1 + 24 * 2)
    if hour == 0:
        ax.annotate('precip real', xy=(0, 0.5), xytext=(-5, 0), xycoords='axes fraction', textcoords='offset points',
                     size='large', ha='right', va='center', rotation='vertical')
    im_precip = plt.imshow(tmean_real[hour], cmap=plt.cm.Blues_r, vmin=0, vmax=1)
    plt.axis('off')
    ax = plt.subplot(4, 24, hour + 1 + 24 * 3)
    if hour == 0:
        ax.annotate('precip gen', xy=(0, 0.5), xytext=(-5, 0), xycoords='axes fraction', textcoords='offset points',
                     size='large', ha='right', va='center', rotation='vertical')
    plt.imshow(tmean_gen[hour], cmap=plt.cm.Blues_r, vmin=0, vmax=1)
    plt.axis('off')
    if hour == 23:
        plt.colorbar()

plt.savefig(f'{plotdir}/distribution_mapplot_{params}_{epoch:04d}.svg')

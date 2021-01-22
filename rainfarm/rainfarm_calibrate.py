"""
based on the rainfarm algorithm from:https://journals.ametsoc.org/view/journals/hydr/7/4/jhm517_1.xml
but adapted.
we dont do any spatial downscaling, only temporal. and the spectral slope for the temporal downscaling
is not based on extrapolating from lower temporal resolution, but on the actual hourly resolution rada data
from the training set.

"""

import pickle
import numpy as np
from skimage.util import view_as_windows

from rainfarm.rainfarm_temporal_downscaling import estimate_alpha ,estimate_beta, downscale_spatiotemporal


startdate = '20090101'
enddate = '20161231'


startdate = '20090101'
enddate = '20161231'

ndomain = 16  # gridpoints
stride = 16
tres = 1

tp_thresh_daily = 5  # mm. in the radardate the unit is mm/h, but then on 5 minutes steps.
# the conversion is done automatically in this script
n_thresh = 20

# load data and precomputed indices
machine = 'kebnekaise'
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

n_samples = len(indices_all)

n_days, nhours, ny, nx = data.shape


## get a set of random samples from the training data

n_calib = 5000

n_repeat = 10
alphas = []
betas= []
for i_repeat in range(n_repeat):
    # get random sample of indices from the precomputed indices
    # for this we generate random indices for the index list (confusing termoonology, since we use
    # indices to index the list of indices...
    ixs = np.random.randint(n_samples, size=n_calib)
    idcs_batch = indices_all[ixs]
    # now we select the data corresponding to these indices

    data_wview = view_as_windows(data, (1, 1, ndomain, ndomain))[..., 0, 0, :,:]
    batch = data_wview[idcs_batch[:, 0], :, idcs_batch[:, 1], idcs_batch[:, 2]]
    # add empty channel dimension (necessary for keras, which expects a channel dimension)
    assert (batch.shape == (n_calib, nhours, ndomain, ndomain))
    assert (~np.any(np.isnan(batch)))

    # remove empty channel

    alpha = estimate_alpha(batch)
    beta = estimate_beta(batch)

    pickle.dump((alpha,beta),open(f'data/spectral_slopes_{i_repeat}.pkl','wb'))
    alphas.append(alpha)
    betas.append(beta)
    print(alpha,beta)

# tests
dsum = batch[0].sum(0)

generated = downscale_spatiotemporal(dsum, alpha, beta, 24)

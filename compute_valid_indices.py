#! /climstorage/sebastian/anaconda3/envs/pr-disagg-env/bin/python
"""

read in formatted data (outut from reformat_data.py),
and determined all valid training samples from it.

valid training samples are all ndomain x ndomain boxes that are free of NaN values,
and where the daily sum exceeds a certain threshold at at least a certain amount of points
(default 5mm on 20 points on a 16x16 domain)

the "sweep" over the domain is controlled by the "stride" parameter. If it is 1, then all possible boxes
are tried out (including those with overlap). with strid=ndomain, all non-overlapping boxes are scanned.

output: .pkl file containing the indices of the training samples

these indices can then be used the following way:
idcs = final_valid_idcs[0]
sub = data[idcs[0],:,idcs[1]:idcs[1]+ndomain,idcs[2]:idcs[2]+ndomain]

@internal: run on misu160

@author: Sebastian Scher
"""

import os
import pickle
import numpy as np
import numba
from dask.diagnostics import ProgressBar

pbar = ProgressBar()
pbar.register()

os.system('mkdir -p data')
# the data is not complete (not all days are available)
# PARAMS
#startdate = '20090101'
#enddate='20091231'
startdate = '20100101'
enddate='20101231'
#enddate = '20161231'
ndomain = 16  # gridpoints
stride = 8  # |ndomain # in which steps to scan the whole domain
tres = 1
tp_thresh_daily = 5  # mm. in the radardate the unit is mm/h, but then on 5 minutes steps.
# the conversion is done automatically in this script
n_thresh = 20
# END PARAMS

if ndomain % 2 != 0:
    raise ValueError(f'ndomain must be an even number')

datapath = '/climstorage/sebastian/pr_disagg/smhi/preprocessed/'
datapath = '/home/s/sebsc/pfs/pr_disagg/smhi_radar/preprocessed'

ifile = f'{datapath}/{startdate}-{enddate}_tres{tres}.npy'

data = np.load(ifile, mmap_mode='r')

if len(data.shape) != 4:
    raise ValueError(f'data has wrong number of dimensions {len(data.shape)} instead of 4')

# compute daily sum, which is the sum over the hour axis
#dsum = np.sum(data, axis=1)
n_days,nhour, ny, nx = data.shape


# compute all valid indices
# for this, we try out all ndomain x ndomain squares shifted by strides, and check whether they have any missing data,
# and if not, whether they adhere to the criteria set by tp_thresh_daily and n_thresh
# since this contains many for loops, we speed it up with numba


@numba.jit
def filter(data):
    final_valid_idcs = []
    # loop over timeslices
    for tidx in numba.prange(n_days):
        print(tidx, '/', n_days)
        # daily sum
        sub = np.nanmean(data[tidx],axis=0)
        # loop over all possible boxes
        for ii in range(0, ny - ndomain, stride):
            for jj in range(0, nx - ndomain, stride):
                subsub = sub[ii:ii + ndomain, jj:jj + ndomain]
                # check for nan values
                if not np.any(np.isnan(subsub)):
                    # if at least n_thresh points are above the threshold,
                    # we use this box
                    if np.sum(subsub > tp_thresh_daily) >= n_thresh:
                        final_valid_idcs.append((tidx, ii, jj))

    return final_valid_idcs


final_valid_idcs = filter(data)

params = f'{startdate}-{enddate}-tp_thresh_daily{tp_thresh_daily}_n_thresh{n_thresh}_ndomain{ndomain}_stride{stride}'
pickle.dump(final_valid_idcs, open(f'data/valid_indices_smhi_radar_{params}.pkl', 'wb'))

print(f'found {len(final_valid_idcs)} valid samples')

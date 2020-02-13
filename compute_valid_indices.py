#! /climstorage/sebastian/anaconda3/envs/pr-disagg-env/bin/python
"""

reard in radar data converted to netcdf, and converr it to npy format suitable for training
in this  step,
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
startdate='20090101'
#enddate='20091231'
enddate='20171231'
ndomain = 16  #gridpoints
stride = 1 # |ndomain # in which steps to scan the whole domain
tres = 1
tp_thresh_daily = 5 # mm. in the radardate the unit is mm/h, but then on 5 minutes steps.
                    # the conversion is done automatically in this script
n_thresh = 20
# END PARAMS

if ndomain %2 != 0:
    raise ValueError(f'ndomain must be an even number')

datapath = '/climstorage/sebastian/pr_disagg/smhi/preprocessed/'

ifile = f'{datapath}/{startdate}-{enddate}_tres{tres}.np.npy'

data = np.load(ifile)

if len(data.shape) !=4:
    raise ValueError(f'data has wrong number of dimensions {len(data.shape)} instead of 4')

# compute daily sum, which is the sum over the hour axis
dsum = np.sum(data,axis=1)
n_days,ny,nx = dsum.shape

# compute all valid indices
# for this, we try out all ndomain x ndomain squares shifted by strides, and check whether they have any missing data,
# and if not, whether they adhere to the criteria set by tp_thresh_daily and n_thresh
# since this contains many for loops, we speed it up with numba

@numba.jit
def filter(dsum):
    final_valid_idcs = []
    # loop over timeslices
    for tidx in numba.prange(n_days):
        print(tidx,'/',n_days)
        sub = dsum[tidx]
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


final_valid_idcs = filter(dsum)

params=f'{startdate}-{enddate}-tp_thresh_daily{tp_thresh_daily}_n_thresh{n_thresh}_ndomain{ndomain}_stride{stride}'
pickle.dump(final_valid_idcs, open(f'data/valid_indices_smhi_radar_{params}.pkl','wb'))

print(f'found {len(final_valid_idcs)} valid samples')

# the indices are used the following way:
# idcs = final_valid_idcs[0]
# sub = data[idcs[0],:,idcs[1]:idcs[1]+ndomain,idcs[2]:idcs[2]+ndomain]

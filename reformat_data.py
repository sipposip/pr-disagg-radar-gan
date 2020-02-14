#! /proj/bolinc/users/x_sebsc/anaconda3/envs/pr-disagg-env/bin/python
#SBATCH -N1
#SBATCH -A snic2019-1-2
#SBATCH -t 1:30:00
#SBATCH --mem=363GB
#! /climstorage/sebastian/anaconda3/envs/pr-disagg-env/bin/python
"""
this script reads in the netcdf radar data (output of convert_smhi_radardata.py),
and converts it to a format suitable for training.

the data is
1) summed to the desired timeresolution "tres" (default 1 hour)
2) reshaped in a format that has hour of the day as separate dimension
    --> output format is (days,tperday,lat,lon)
3) saved as a single .npz file


note that this script is not very memory efficient. if you dont have enough RAM,
then it would be better to process each year individually

@internal: run on tetralith

@author: Sebastian Scher
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar

pbar = ProgressBar()
pbar.register()


#PARAMS
startdate='20090101'
#startdate='20100101'
#enddate='20091231'
enddate='20161231'
#enddate='20101231'
tres=1 # [h]

# END PARAMS
# the radardata is 5 minute data, but in mm/h. so to get mm/day for the daysums,
# we need to divide by 60/5=12
conv_factor = 1/12

#datapath='/climstorage/sebastian/pr_disagg/smhi/netcdf/'
datapath='/proj/bolinc/users/x_sebsc/pr_disagg/smhi/netcdf/'

#outpath='/climstorage/sebastian/pr_disagg/smhi/preprocessed/'
outpath='/proj/bolinc/users/x_sebsc/pr_disagg/smhi/preprocessed/'
os.system(f'mkdir -p {outpath}')

# create list of available files
dates_all = pd.date_range(startdate,enddate,freq='1d')
ifiles = []
for date in dates_all:
    fname = f'{datapath}/smhi_radar_{date.strftime("%Y%m%d")}.nc'
    if os.path.exists(fname):
        ifiles.append(fname)

if len(ifiles) == 0:
    raise Exception('no input files found!')


# now open all files lazily
# they are automatically chunked per file (thus per day)
data_raw = xr.open_mfdataset(ifiles, combine='nested', concat_dim='time')
data_raw = data_raw['__xarray_dataarray_variable__']
# convert to 32bit
data_raw = data_raw.astype('float32')

# sum to desired timeresolution
agg = data_raw.resample(time=f'{tres}h', label='left').sum(skipna=False)
# convert to numpy array
agg = agg.values

# now we want to reshape to (days,tperday,lat,lon)
t_per_day = int(24/tres)

ntime,ny,nx = agg.shape
ndays = ntime / t_per_day
assert(ndays.is_integer())
ndays = int(ndays)
reshaped = agg.reshape((ndays,t_per_day,ny,nx))

final = reshaped

np.savez_compressed(f'{outpath}/{startdate}-{enddate}_tres{tres}.npz',data=final)
np.save(f'{outpath}/{startdate}-{enddate}_tres{tres}', final)

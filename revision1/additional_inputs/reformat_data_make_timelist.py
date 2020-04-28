#! /proj/bolinc/users/x_sebsc/anaconda3/envs/pr-disagg-env/bin/python
#SBATCH -N1
#SBATCH -A snic2019-1-2
#SBATCH -t 1:30:00


import os
import numpy as np
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar

pbar = ProgressBar()
pbar.register()


#PARAMS
# for training data:
# startdate = '20090101'
# enddate = '20161231'
# for test data:
startdate = '20170101'
enddate = '20181231'
tres=1 # [h]

# END PARAMS
# the radardata is 5 minute data, but in mm/h. so to get mm/day for the daysums,
# we need to divide by 60/5=12
conv_factor = 1/12

datapath='/proj/bolinc/users/x_sebsc/pr_disagg/smhi/netcdf/'

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

time_daily = agg.time[::int(24/tres)]

doy = time_daily.dt.dayofyear.values

np.save(f'{outpath}/{startdate}-{enddate}_tres{tres}_doy', doy)

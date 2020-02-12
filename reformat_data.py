#! /climstorage/sebastian/anaconda3/envs/pr-disagg-env/bin/python

import os
import numpy as np
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar

pbar = ProgressBar()
pbar.register()



startdate='20090101'
enddate='20091231'
#enddate='20171231'
tres=1 # [h]

# END PARAMS
# the radardata is 5 minute data, but in mm/h. so to get mm/day for the daysums,
# we need to divide by 60/5=12
conv_factor = 1/12

datapath='/climstorage/sebastian/pr_disagg/smhi/netcdf/'

outpath='/climstorage/sebastian/pr_disagg/smhi/preprocessed/'
os.system(f'mkdir -p {outpath}')

# create list of available filies
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

# # replace missing vals with nan
# data = data_raw.where(data_raw != 255)

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

# convert to 32bit
final = final.astype('float32')

np.save(f'{outpath}/{startdate}-{enddate}_tres{tres}.np',final)




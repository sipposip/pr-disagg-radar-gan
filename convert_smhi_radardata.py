#! /climstorage/sebastian/anaconda3/envs/pr-disagg-env/bin/python
"""
convert the single .tif smhi radar files to daily netcdf files

"""

import os
from tqdm import tqdm
import xarray as xr
import pandas as pd

path='/climstorage/sebastian/pr_disagg/smhi/'
outpath = f'{path}/netcdf/'
os.system(f'mkdir -p {outpath}')


dates = pd.date_range('20090101','20191231')
failed_dates = [] # the radar data is not complete
for date in tqdm(dates):
    # the smhi radar has 5 minute timesteps
    try:
        res = []
        for hour in range(0, 24):
            for minute in range(0, 60, 5):
                iname = f'{path}/radar_{date.strftime("%y%m%d")}{hour:02}{minute:02}.tif'
                da = xr.open_rasterio(iname)
                da['time'] = pd.to_datetime(f'{date.strftime("%Y%m%d")}{hour:02}{minute:02}')
                res.append(da)

        res = xr.concat(res, dim='time')
        if res['band'].size >1:
             # select first band (for data from 2015-2017 there are 3 bands, but quick
             # inspeciton showed that they seem to have the same data
            res = res.isel(band=0)
        # remove empty band dimension
        res = res.squeeze()

        # replace missing vals with nan (missing value is 255)
        res = res.where(res != 255)
        # convert to mm/5mins (https://opendata.smhi.se/apidocs/radar/data.html)
        dbz = res * 0.4 - 30
        mmperh = ((10 ** (dbz / 10)) / 200) ** (1 / 1.5)
        mmper5min = mmperh * 5/60
        res = mmper5min
        res.to_netcdf(f'{outpath}/smhi_radar_{date.strftime("%Y%m%d")}.nc')

    except:
        print(f'date {date} failed, skipping')
        failed_dates.append(date)

print('failed_dates:')
print(failed_dates)

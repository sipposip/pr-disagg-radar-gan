import pandas as pd
import os

outpath='/climstorage/sebastian/pr_disagg/smhi/'
os.system(f'mkdir -p {outpath}')
for date in pd.date_range('20090101','20191231'):
    url = f'https://opendata-download-radar.smhi.se/api/version/latest/area/sweden/product/comp/{date.year}/{date.month}/{date.day}.zip?format=tif'
    outname = f'{outpath}/smhi_radar_{date.strftime("%Y%m%d")}.zip'
    os.system(f'wget -O {outname} {url}')


os.system(' for f in smhi_radar*.zip; do unzip -o ${f}; rm ${f}; done')

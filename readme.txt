



download_smhi_radardata.py downloads the raw radar data in geotiff format

convert_smhi_radardata.py converts the raw radardate to daily netcdf files, including conversion
from radar reflectivities to precipitation

reformat_data.py reads in the netcdf data, aggregates it to desired aggregation (default 1h), and
outputs it in a single .npy file in a format suitable for machine learning


compute_valid_indices.py reads in the data from reformat_data.py and calculates all possible
indices that have (1) no missing data an (2) pass a selection criterion for enough rainfall.
it outputs a .pkl file with the indices



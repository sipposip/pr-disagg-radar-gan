RainDisaggGAN is a method that uses GANs for temporarly downsampling spatial precipitation patterns.

The method is described in our paper (INSERT LINK)


This repository contains the trained network, and an example script on how to use it for temporaral disaggregation.


Additionally the repository contains all necessary scripts to fully replicate the study:

download_smhi_radardata.py downloads the raw radar data in geotiff format

convert_smhi_radardata.py converts the raw radardate to daily netcdf files, including conversion
from radar reflectivities to precipitation

reformat_data.py reads in the netcdf data, aggregates it to desired aggregation (default 1h), and
outputs it in a single .npy file in a format suitable for machine learning

compute_valid_indices.py reads in the data from reformat_data.py and calculates all possible
indices that have (1) no missing data an (2) pass a selection criterion for enough rainfall.
it outputs a .pkl file with the indices

gan_train_cwgangp_pixelnorm.py  trains the GAN, and makes some intermediate plots

generate_and_evaluate.py evaluates the GAN and makes analysis plots

The GAN is built and trained with Tensorflow 2.1

Note that all datapaths in the scripts need to be adapted to your local system
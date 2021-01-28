
import matplotlib
matplotlib.use('agg')
import pickle
import os
from tqdm import tqdm, trange
import numpy as np
import properscoring
from rainfarm.rainfarm_temporal_downscaling import downscale_spatiotemporal

plotdir='plots_generated_rainfarm'
os.system(f'mkdir -p {plotdir}')

reals = np.load('/climstorage/sebastian/pr_disagg/data/real_samples.npy')
reals_dsum = np.sum(reals,axis=1)

alpha, beta = pickle.load(open('data/spectral_slopes.pkl','rb'))


# compute statistics over
# many generated smaples
# we compute the areamean,
n_sample = 1000
n_fake_per_real = 1000

crps_amean_all_rainfarm = []
for i in trange(n_sample):
    real = reals[i]
    dsum = reals_dsum[i]

    generated = np.array([downscale_spatiotemporal(dsum, alpha, beta, 24) for p in range(n_fake_per_real)])
    crps = properscoring.crps_ensemble(real, generated, axis=0)
    crps_amean_all_rainfarm.append(crps)

crps_amean_all_rainfarm = np.array(crps_amean_all_rainfarm)
pickle.dump((crps_amean_all_rainfarm), open('data/crps_results_rainfarm.pkl','wb'))

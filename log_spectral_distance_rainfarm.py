"""

adapted from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/

"""

import pickle
import numpy as np
from scipy import fftpack
from tqdm import tqdm, trange
import numba
import seaborn as sns

from pylab import plt

def azimuthal_verage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def compute_radial_spectrum(x):
    # 2d ffft
    f1 = fftpack.fft2(x)
    f2 = fftpack.fftshift(f1)
    psd2D = np.abs(f2) ** 2
    psd1D = azimuthal_verage(psd2D)
    return psd1D


@numba.njit(parallel=True)
def log_spectral_distance(ps1, ps2):
    """compute the log spectral distance between
    two power spectra
    the integral over the frequency is approximated via a sum.
    """
    n_omega = len(ps1)
    assert (len(ps2) == n_omega)
    return np.sqrt(np.sum((10 * np.log10(ps1 / ps2)) ** 2)) * 1 / n_omega


generated_rainfarm = np.load('/climstorage/sebastian/pr_disagg/data/generated_samples_rainfarm.npy')


n_samples = 1000
generated = generated_rainfarm[:n_samples]


# flatten the hour dimension into the sample dimension

generated_flathour = generated.reshape(-1,*generated.shape[2:])

spectra_gen = np.array([compute_radial_spectrum(x) for x in tqdm(generated_flathour)])


n = len(spectra_gen)

@numba.njit(parallel=True)
def compute_dists(x,y):

    n=len(x)
    dists = np.zeros(n**2) # this needs too much memory!!!
    for i in range(n):
        print(i,n)
        for j in range(n):
            if i != j:
                d = log_spectral_distance(x[i], y[j])
                dists[i*n+j] = d

    return dists


dists_gen = compute_dists(spectra_gen, spectra_gen)
# remove diagonal elements (they are zero)
indices = np.arange(n ** 2)
diags = np.arange(0, n ** 2, n)
indices_no_diag = [e for e in indices if e not in diags]
dists_gen = dists_gen[indices_no_diag]

res = {'gen_rainfarm':dists_gen}
pickle.dump(res,open('log_spectral_distances.pkl','wb'))

sns.set_palette('colorblind')

plt.figure()
sns.kdeplot(dists_gen, label='generated rainfarm')
plt.xlabel('log spectral distance')
plt.legend()
sns.despine()
plt.savefig(f'plots/log_spectral_distances_n{n_samples}_rainfarm.svg')


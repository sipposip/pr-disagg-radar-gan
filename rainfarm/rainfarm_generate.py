
import matplotlib
matplotlib.use('agg')
import pickle
import os
from tqdm import tqdm
import numpy as np
from pylab import plt
from matplotlib.colors import LogNorm
import seaborn as sns

from rainfarm.rainfarm_temporal_downscaling import downscale_spatiotemporal

plotdir='plots_generated_rainfarm'
os.system(f'mkdir -p {plotdir}')

reals = np.load('/climstorage/sebastian/pr_disagg/data/real_samples.npy')
reals_dsum = np.sum(reals,axis=1)

alpha, beta = pickle.load(open('data/spectral_slopes.pkl','rb'))

# for a start: generate one artificial for each real one
generated = np.array([downscale_spatiotemporal(p,alpha,beta,24) for p in tqdm(reals_dsum)])

np.save('/climstorage/sebastian/pr_disagg/data/generated_samples_rainfarm.npy',generated)

amean_gen = np.mean(generated, axis=(1,2))
amean_real = np.mean(reals, axis=(1,2))



def ecdf(data):
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x, y)


sns.set_palette('colorblind')
# ecdf of area means. the hours are flattened
plt.figure()
ax1 = plt.subplot(211)
plt.plot(*ecdf(amean_gen.flatten()), label='gen')
plt.plot(*ecdf(amean_real.flatten()), label='real')
plt.legend(loc='upper left')
sns.despine()
plt.xlabel('mm/h')
plt.ylabel('ecdf areamean')
plt.semilogx()
# ecdf of (flattened) spatial data
ax2 = plt.subplot(212)
plt.plot(*ecdf(generated.flatten()), label='gen')
plt.plot(*ecdf(reals.flatten()), label='real')
plt.legend(loc='upper left')
sns.despine()
plt.ylabel('ecdf')
plt.xlabel('mm/h')
plt.semilogx()
plt.tight_layout()
plt.savefig(f'{plotdir}/ecdf_allx_rainfarm.png', dpi=400)
# cut at 0.1mm/h
ax1.set_xlim(xmin=0.5)
ax1.set_ylim(ymin=0.8, ymax=1.01)
ax2.set_xlim(xmin=0.1)
ax2.set_ylim(ymin=0.6, ymax=1.01)
plt.savefig(f'{plotdir}/ecdf_rainfarm.png', dpi=400)


plt.rcParams['savefig.bbox'] = 'tight'
cmap = plt.cm.gist_earth_r
plotnorm = LogNorm(vmin=0.01, vmax=50)

n_to_generate = 20
n_fake_per_real = 15
n_plot = n_fake_per_real + 1
plotcount=0
for i in range(n_to_generate):
    real, dsum = reals[i],reals_dsum[i]
    plotcount += 1

    generated = np.array([downscale_spatiotemporal(dsum,alpha,beta,24) for _ in range(n_fake_per_real)])

    # now the same, but showing absolute precipitation fields
    # compute absolute precipitation from fraction of daily sum.
    # this can be done with numpy broadcasting.
    # we also have to multiply with norm_scale (because cond is normalized)
    generated_scaled = generated
    real_scaled = real
    fig = plt.figure(figsize=(25, 12))
    # plot real one
    ax = plt.subplot(n_plot, 25, 1)
    im = plt.imshow(dsum, cmap=cmap, norm=plotnorm)
    plt.axis('off')
    ax.annotate('real', xy=(0, 0.5), xytext=(-5, 0), xycoords='axes fraction', textcoords='offset points',
                size='large', ha='right', va='center', rotation='vertical')
    ax.annotate(f'daily sum', xy=(0.5, 1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

    for jplot in range(1, 24 + 1):
        ax = plt.subplot(n_plot, 25, jplot + 1)
        plt.imshow(real_scaled[jplot - 1, :, :].squeeze(), cmap=cmap, norm=plotnorm)
        plt.axis('off')
        ax.annotate(f'{jplot:02d}'':00', xy=(0.5, 1), xytext=(0, 5),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    # plot fake samples
    for iplot in range(n_fake_per_real):
        plt.subplot(n_plot, 25, (iplot + 1) * 25 + 1)
        plt.imshow(dsum, cmap=cmap, norm=plotnorm)
        plt.axis('off')
        for jplot in range(1, 24 + 1):
            plt.subplot(n_plot, 25, (iplot + 1) * 25 + jplot + 1)
            plt.imshow(generated_scaled[iplot, jplot - 1, :, :].squeeze(), cmap=cmap, norm=plotnorm)
            plt.axis('off')
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.007, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('precipitation [mm]', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    plt.savefig(f'{plotdir}/generated_precip_rainfarm_{plotcount:04d}.png')

    plt.close('all')


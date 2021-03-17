import json
import pickle
import numpy as np
from scipy import stats

gan, random = pickle.load(open(f'data/crps_results_n_sample10000.pkl', 'rb'))
rainfarm = pickle.load(open(f'data/crps_results_rainfarm.pkl', 'rb'))

print('gan', gan.mean())
print('random', random.mean())
print('rainfarm', rainfarm.mean())

# 1 sample ttest
_, p = stats.ttest_1samp((gan - random).flatten(), popmean=0)
print(p)

res = {'gan': gan.mean(),
       'random': random.mean(),
       'rainfarm': rainfarm.mean()
       }

json.dump(res, open('data/crps_results.json', 'w'))


def bootstrapped_difference_onesample(x1, perc=1, N=10000):
    """ compute difference between x1 and x2 plus uncertainty, when x1 and x2 are either rmse or standarddeviation
    """
    n_samples = len(x1)
    means = []
    for i in range(N):
        indices = np.random.choice(n_samples, replace=True, size=n_samples)
        # now compute difference in  RMSE on this subsample
        mm = np.mean(x1[indices])
        means.append(mm)
    means = np.array(means)
    mmean = np.mean(x1)
    upper = np.percentile(means, q=100 - perc)
    lower = np.percentile(means, q=perc)
    # assert (upper >= lower) # we deactivate this check here because if one or both of x1 and x2
    # concist only of repreated values, then numerical inaccuracis can lead to
    # lower being a tiny little larger than upper (even though they should be the same in this case)
    return np.array([mmean, lower, upper])


bootstrap_res = bootstrapped_difference_onesample((gan - random).flatten())

print(bootstrap_res)

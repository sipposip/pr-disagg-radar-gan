
import json
import pickle
import numpy as np
from scipy import stats

gan, random = pickle.load(open(f'data/crps_results_n_sample10000.pkl', 'rb'))
rainfarm = pickle.load(open(f'data/crps_results_rainfarm.pkl', 'rb'))


print('gan',gan.mean())
print('random',random.mean())
print('rainfarm',rainfarm.mean())

# 1 sample ttest
_,p=stats.ttest_1samp((gan-random).flatten(), popmean=0)
print(p)

res = {'gan':gan.mean(),
       'random':random.mean(),
       'rainfarm':rainfarm.mean()
       }


json.dump(res,open('data/crps_results.json','w'))

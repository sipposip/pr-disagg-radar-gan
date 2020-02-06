#! /climstorage/sebastian/anaconda3/envs/pr-disagg-env/bin/python
"""


"""
import pickle
import numpy as np
import tensorflow as tf

startdate='20090101'
enddate='20091231'
#enddate='20171231'
ndomain = 16  #gridpoints
tres = 1
tp_thresh_daily = 5 # mm. in the radardate the unit is mm/h, but then on 5 minutes steps.
                    # the conversion is done automatically in this script
n_thresh = 20


# load data and precomputed indices
print('loading data')
converted_data_path = '/climstorage/sebastian/pr_disagg/smhi/preprocessed/'
indices_data_path = './data/'

data_ifile = f'{converted_data_path}/{startdate}-{enddate}_tres{tres}.np.npy'

params=f'{startdate}-{enddate}-tp_thresh_daily{tp_thresh_daily}_n_thresh{n_thresh}_ndomain{ndomain}'
indices_file = f'data/valid_indices_smhi_radar_{params}.pkl'

data = np.load(data_ifile)

indices_all = pickle.load(open(indices_file,'rb'))
# convert to array
indices_all = np.array(indices_all)
# this has shapes (nsamples,3)
# each ros is (tidx,yidx,xidx)
print('finished loading data')

# the data has dimensions (sample,hourofday,x,y)
n_days,nhours,ny,nx = data.shape
# sanity checks
assert(len(data.shape)==4)
assert(len(indices_all.shape)==2)
assert(indices_all.shape[1]==3)
assert(nhours==24//tres)
assert(np.max(indices_all[:,0]) < n_days)
assert(np.max(indices_all[:,1]) < ny)
assert(np.max(indices_all[:,2]) < nx)
assert(data.dtype=='float32')

n_samples = len(indices_all)


# compute daily sum as condition
dsum = data.sum(axis=1)


# normalization
# the daily sum we normalize so that the 90th percentile is 1
norm_scale = np.nanpercentile(dsum,90)

dsum = dsum * norm_scale

# convert the subdaily data to fractions of the daily sum
fractions = data.copy()
for i in range(n_days):
    fractions[i] = data[i] / data[i].sum(axis=0) # sum over day

assert(np.nanmax(fractions)<=1)
assert(np.nanmin(fractions)>=0)

latent_dim = 1024


def generate_real_samples(n_batch):

    # get random indices from the precomputed indices
    # for this we generate random indices for the index list (confusing termoonology, since we use
    # indices to index the list of indices...
    ixs = np.random.choice(np.arange(n_samples), size=n_batch, replace=False)
    idcs_batch = indices_all[ixs]

    batch = np.empty((n_batch,nhours,ndomain,ndomain), dtype='float32')
    batch_cond = np.empty((n_batch,ndomain,ndomain), dtype='float32')
    #TODO: the following implementation is potentially quite slow
    # either get it working with fancy indexing, or use numba
    for i in range(n_batch):
        tidx, iy,ix = idcs_batch[i]
        batch[i,:,:,:] = fractions[tidx, :, iy:iy+ndomain, ix:ix+ndomain]
        batch_cond[i,:,:] = dsum[tidx, iy:iy+ndomain, ix:ix+ndomain]

    return [batch, batch_cond]
#
# # generate points in latent space as input for the generator


def generate_latent_points(n_batch):

    # generate points in the latent space
    latent = np.random.normal(size=(n_batch, latent_dim))
    # randomly select conditions
    ixs = np.random.choice(np.arange(n_samples), size=n_batch, replace=False)
    idcs_batch = indices_all[ixs]
    batch_cond = np.empty((n_batch, ndomain, ndomain), dtype='float32')
    for i in range(n_batch):
        tidx, iy,ix = idcs_batch[i]
        batch_cond[i,:,:] = dsum[tidx, iy:iy+ndomain, ix:ix+ndomain]
    return [latent, batch_cond]

#
# # use the generator to generate n fake examples, with class labels
# def generate_fake_samples(generator, n_samples):
#     # generate points in latent space
#     cond_in, latent_in = generate_latent_points(latent_dim, n_samples)
#     # predict outputs
#     # images = generator.predict([cond_in, latent_in])
#     images = generator.predict(latent_in)
#     return [images, cond_in]
#
#
# def generate(cond):
#     latent = np.random.normal(size=(1, latent_dim))
#     cond = np.expand_dims(cond,0)
#     # return  gen.predict([cond, latent])
#     return  gen.predict(latent)
#
#
#


def wasserstein_loss(y_true, y_pred):
    # we use -1 for fake, and +1 for real labels
    return tf.reduce_mean(y_true * y_pred)


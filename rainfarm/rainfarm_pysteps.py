# -*- coding: utf-8 -*-
"""
pysteps.downscaling.rainfarm
============================
Implementation of the RainFARM stochastic downscaling method as described in
:cite:`Rebora2006`.
.. autosummary::
    :toctree: ../generated/
    downscale
"""

import warnings

import numpy as np
from scipy.ndimage import convolve


def _log_slope(log_k, log_power_spectrum):
    lk_min = log_k.min()
    lk_max = log_k.max()
    lk_range = lk_max - lk_min
    lk_min += (1 / 6) * lk_range
    lk_max -= (1 / 6) * lk_range

    selected = (lk_min <= log_k) & (log_k <= lk_max)
    lk_sel = log_k[selected]
    ps_sel = log_power_spectrum[selected]
    alpha = np.polyfit(lk_sel, ps_sel, 1)[0]
    alpha = -alpha

    return alpha


def _balanced_spatial_average(x, k):
    ones = np.ones_like(x)
    return convolve(x, k) / convolve(ones, k)


def downscale(precip, alpha=None, ds_factor=16, threshold=None, return_alpha=False):
    """
    Downscale a rainfall field by a given factor.
    Parameters
    ----------
    precip: array_like
        Array of shape (m,n) containing the input field.
        The input is expected to contain rain rate values.
    alpha: float, optional
        Spectral slope. If none, the slope is estimated from
        the input array.
    ds_factor: int, optional
        Downscaling factor.
    threshold: float, optional
        Set all values lower than the threshold to zero.
    return_alpha: bool, optional
        Whether to return the estimated spectral slope `alpha`.
    Returns
    -------
    r: array_like
        Array of shape (m*ds_factor,n*ds_factor) containing
        the downscaled field.
    alpha: float
        Returned only when `return_alpha=True`.
    Notes
    -----
    Currently, the pysteps implementation of RainFARM only covers spatial downscaling.
    That is, it can improve the spatial resolution of a rainfall field. However, unlike
    the original algorithm from Rebora et al. (2006), it cannot downscale the temporal
    dimension.
    References
    ----------
    :cite:`Rebora2006`
    """

    ki = np.fft.fftfreq(precip.shape[0])
    kj = np.fft.fftfreq(precip.shape[1])
    k_sqr = ki[:, None] ** 2 + kj[None, :] ** 2
    k = np.sqrt(k_sqr)

    ki_ds = np.fft.fftfreq(precip.shape[0] * ds_factor, d=1 / ds_factor)
    kj_ds = np.fft.fftfreq(precip.shape[1] * ds_factor, d=1 / ds_factor)
    k_ds_sqr = ki_ds[:, None] ** 2 + kj_ds[None, :] ** 2
    k_ds = np.sqrt(k_ds_sqr)


    # compute spectral slope
    if alpha is None:
        fp = np.fft.fft2(precip)
        fp_abs = abs(fp)
        log_power_spectrum = np.log(fp_abs ** 2)
        valid = (k != 0) & np.isfinite(log_power_spectrum)
        alpha = _log_slope(np.log(k[valid]), log_power_spectrum[valid])




    # create random field with size of disaggregated field
    fg = np.exp(complex(0, 1) * 2 * np.pi * np.random.rand(*k_ds.shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # for this we need an extensio of k_ds_sqr that also contains omega
        fg *= np.sqrt(k_ds_sqr ** (-alpha / 2))
    fg[0, 0] = 0
    g = np.fft.ifft2(fg).real
    g /= g.std()
    r = np.exp(g)

    # normalize the field on a point by point bases so that it has the same properties
    # as the original field on all scales that are covered by the original field.
    P_u = np.repeat(np.repeat(precip, ds_factor, axis=0), ds_factor, axis=1)
    rad = int(round(ds_factor / np.sqrt(np.pi)))
    (mx, my) = np.mgrid[-rad : rad + 0.01, -rad : rad + 0.01]
    tophat = ((mx ** 2 + my ** 2) <= rad ** 2).astype(float)
    tophat /= tophat.sum()

    P_agg = _balanced_spatial_average(P_u, tophat)
    r_agg = _balanced_spatial_average(r, tophat)
    r *= P_agg / r_agg

    if threshold is not None:
        r[r < threshold] = 0

    if return_alpha:
        return r, alpha

    return r


def downscale_spatiotemporal(precip, alpha , beta,ds_t_factor=1):
    """
    donwscales from empty time dimension to ds_t_factor
    """


    ki = np.fft.fftfreq(precip.shape[0])
    kj = np.fft.fftfreq(precip.shape[1])
    k_sqr = ki[:, None] ** 2 + kj[None, :] ** 2
    k = np.sqrt(k_sqr)

    om = 2*np.pi*np.fft.fftfreq(ds_t_factor)
    # we need om in complex form (for equation (2) in the rainfarm paper).
    # this is because we have negative frequencies, and we take roots of these
    # frequencies, so the result is complex.
    om = om.astype(np.complex)
    ni, nj = k.shape
    n_t = len(om)

    # create random field with size of disaggregated field
    # todo: whey 2pi? this came from pysteps, but i cannot find this n the rainfarm paper....
    phi = np.exp(complex(0, 1) * 2 * np.pi * np.random.rand(n_t,ni,nj))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # change to incorporate omega and beta (eq 2 in paper)
        # for this we need to exoand (repeat) k_sqr to the omega dimension
        # and om needs to extended to the spatial dimension.
        # we don't need to dothid explicitely, we can use broadcasting instead.
        # dot this we need to swap the order of om and k_sqrt (mathematically this does not make a difference,
        # but it is necessary for the numpmy broadcasting rules
        fg = phi * np.sqrt(om[:,None,None]**(-beta)*k_sqr[None,:,:] ** (-alpha / 2))
    # everything at positions with one of the wavenumber zero is nan or inf. set to zero instead.
    fg[0] = 0
    fg[:,0,0] = 0
    assert(np.all(np.isfinite(fg)))
    g = np.fft.ifft2(fg).real
    g /= g.std()
    r = np.exp(g)

    # normalize the field on a point by point bases so that it has the same properties
    # as the original field on all scales that are covered by the original field.
    # in our case this simply means that the sum over time at each gridpoint is the same
    # as the original one

    r_tsum = np.sum(r,axis=0)
    r = r * precip/r_tsum

    return r

## tests

# precip = np.random.random((10,20))
#
# r = downscale(precip, ds_factor=2)
#
# precip = np.random.random((10,20,24))
#
# r = downscale_spatiotemporal(precip, ds_factor=2, ds_t_factor=2)
#
# # open questions:
# # should we do the rainfarm thing on the whole domain, or on our 16x16 subsets?
# # whole domain is tricky due to missing data....
#
#
#
# # scratch
#
# log_k, log_power_spectrum = np.log(om_fullgrid[valid]), log_power_spectrum[valid]
#

def estimate_beta(p_samples):
    """
        estimate temporal spectral slope from a list of samples
        with each sample in the from (t_sub,lat,lon)
    :param p_samples:
    :return: beta
    """

    n_samples = p_samples.shape[0]
    # n_t is the number of timesteps in each sample
    n_t, ni, nj = p_samples.shape[1:]

    # compute power spectrum along the time axis
    fp = np.fft.fft(p_samples,axis=1)
    fp_abs = abs(fp)
    log_power_spectrum = np.log(fp_abs ** 2)
    # compute wavenumbers
    om = 2 * np.pi * np.fft.fftfreq(n_t)
    # convert to absolute
    om = np.sqrt(om**2)
    # extend om to the fullgrid of p_samples
    om_fullgrid = np.repeat(np.repeat(np.repeat(om[:,None], ni, axis=1)[:, :,None], nj, axis=2)[None,:,:,:],
                            n_samples,axis=0)

    assert(om_fullgrid.shape==p_samples.shape)
    # now select all valid om values and the corresponding values from the power spectrum.
    # in this step, the data is automatically flattened and can be put in the slope
    # estimation routine
    valid = (om_fullgrid != 0) & np.isfinite(log_power_spectrum)
    beta = _log_slope(np.log(om_fullgrid[valid]), log_power_spectrum[valid])
    return beta


def estimate_alpha(p_samples):
    """
        estimate spatial spectral slope from a list of samples
        with each sample in the from (t_sub,lat,lon)
    :param p_samples:
    :return: beta
    """
    n_samples = p_samples.shape[0]
    # n_t is the number of timesteps in each sample
    n_t,ni,nj = p_samples.shape[1:]

    # compute power spectrum along the two spatial axes
    fp = np.fft.fftn(p_samples,axes=(2,3))
    fp_abs = abs(fp)
    log_power_spectrum = np.log(fp_abs ** 2)
    # compute wavenumbers
    ki = np.fft.fftfreq(ni)
    kj = np.fft.fftfreq(nj)
    k_sqr = ki[:, None] ** 2 + kj[None, :] ** 2
    k = np.sqrt(k_sqr)
    k_fullgrid = np.repeat(np.repeat(k[None,:, :], n_t, axis=0)[None,:,:,:],n_samples,axis=0)

    assert(k_fullgrid.shape==p_samples.shape)
    # now select all valid om values and the corresponding values from the power spectrum.
    # in this step, the data is automatically flattened and can be put in the slope
    # estimation routine
    valid = (k_fullgrid != 0) & np.isfinite(log_power_spectrum)
    alpha = _log_slope(np.log(k_fullgrid[valid]), log_power_spectrum[valid])
    return alpha


p_samples = np.random.random((5,24,4,6))
alpha = estimate_alpha(p_samples)
beta = estimate_beta(p_samples)


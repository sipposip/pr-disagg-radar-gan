import warnings

import numpy as np


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


def estimate_beta(p_samples):
    """
        estimate temporal spectral slope from a list of samples
        with each sample in the from (t_sub,lat,lon)
        p_samples: [n_sample,n_t_sub,nlat,nlon]
    """

    n_samples = p_samples.shape[0]
    # n_t is the number of timesteps in each sample
    n_t, ni, nj = p_samples.shape[1:]

    # compute power spectrum along the time axis
    fp = np.fft.fft(p_samples, axis=1)
    fp_abs = abs(fp)
    log_power_spectrum = np.log(fp_abs ** 2)
    # compute wavenumbers
    om = 2 * np.pi * np.fft.fftfreq(n_t)
    # convert to absolute
    om = np.sqrt(om ** 2)
    # extend om to the fullgrid of p_samples
    om_fullgrid = np.repeat(np.repeat(np.repeat(om[:, None], ni, axis=1)[:, :, None], nj, axis=2)[None, :, :, :],
                            n_samples, axis=0)

    assert (om_fullgrid.shape == p_samples.shape)
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
        p_samples: [n_sample,n_t_sub,nlat,nlon]
    """
    n_samples = p_samples.shape[0]
    # n_t is the number of timesteps in each sample
    n_t, ni, nj = p_samples.shape[1:]

    # compute power spectrum along the two spatial axes
    fp = np.fft.fftn(p_samples, axes=(2, 3))
    fp_abs = abs(fp)
    log_power_spectrum = np.log(fp_abs ** 2)
    # compute wavenumbers
    ki = np.fft.fftfreq(ni)
    kj = np.fft.fftfreq(nj)
    k_sqr = ki[:, None] ** 2 + kj[None, :] ** 2
    k = np.sqrt(k_sqr)
    k_fullgrid = np.repeat(np.repeat(k[None, :, :], n_t, axis=0)[None, :, :, :], n_samples, axis=0)

    assert (k_fullgrid.shape == p_samples.shape)
    # now select all valid om values and the corresponding values from the power spectrum.
    # in this step, the data is automatically flattened and can be put in the slope
    # estimation routine
    valid = (k_fullgrid != 0) & np.isfinite(log_power_spectrum)
    alpha = _log_slope(np.log(k_fullgrid[valid]), log_power_spectrum[valid])
    return alpha


def downscale_spatiotemporal(precip, alpha, beta, ds_t_factor):
    """
    donwscales from empty time dimension to ds_t_factor
    """

    ki = np.fft.fftfreq(precip.shape[0])
    kj = np.fft.fftfreq(precip.shape[1])
    k_sqr = ki[:, None] ** 2 + kj[None, :] ** 2
    k = np.sqrt(k_sqr)

    om = 2 * np.pi * np.fft.fftfreq(ds_t_factor)
    # we need om in complex form (for equation (2) in the rainfarm paper).
    # this is because we have negative frequencies, and we take roots of these
    # frequencies, so the result is complex.
    om = om.astype(np.complex)
    ni, nj = k.shape
    n_t = len(om)

    # create random field with size of disaggregated field
    phi = np.exp(complex(0, 1) * 2 * np.pi * np.random.rand(n_t, ni, nj))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # change to incorporate omega and beta (eq 2 in paper)
        # for this we need to exoand (repeat) k_sqr to the omega dimension
        # and om needs to extended to the spatial dimension.
        # we don't need to do this explicitely, we can use broadcasting instead.
        # dot this we need to swap the order of om and k_sqrt (mathematically this does not make a difference,
        # but it is necessary for the numpmy broadcasting rules
        fg = phi * np.sqrt((om[:, None, None] ** (-beta)) * k_sqr[None, :, :] ** (-alpha / 2))
    # everything at positions with one of the wavenumber zero is nan or inf. set to zero instead.
    fg[0] = 0
    fg[:, 0, 0] = 0
    assert (np.all(np.isfinite(fg)))
    g = np.fft.ifft2(fg).real
    g /= g.std()
    r = np.exp(g)
    # normalize the field on a point by point bases so that it has the same properties
    # as the original field on all scales that are covered by the original field.
    # in our case this simply means that the sum over time at each gridpoint is the same
    # as the original one
    r_tsum = np.sum(r, axis=0)
    r = r * precip / r_tsum

    return r


# tres = 24
# nlat, nlon = 4, 6
# p_samples = np.random.random((5, tres, nlat, nlon))
# alpha = estimate_alpha(p_samples)
# beta = estimate_beta(p_samples)
# p_daily = np.random.random((nlat, nlon))
#
# p_downscaled = downscale_spatiotemporal(p_daily, alpha, beta, ds_t_factor=tres)

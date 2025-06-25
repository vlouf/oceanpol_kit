"""
Compute the attenuation

@author: Valentin Louf
@email: valentin.louf@bom.gov.au
@institution: Bureau of Meteorology
@date: 19/03/2025

.. autosummary::
    :toctree: generated/

    smooth_and_trim
    correct_attenuation
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid


def smooth_and_trim(x, window_len=11, window="hanning"):
    """
    Smooth data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x : array
        The input signal.
    window_len : int, optional
        The dimension of the smoothing window; should be an odd integer.
    window : str
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman' or 'sg_smooth'. A flat window will produce a moving
        average smoothing.

    Returns
    -------
    y : array
        The smoothed signal with length equal to the input signal.

    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    valid_windows = ["flat", "hanning", "hamming", "bartlett", "blackman", "sg_smooth"]
    if window not in valid_windows:
        raise ValueError("Window is on of " + " ".join(valid_windows))

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]

    if window == "flat":  # moving average
        w = np.ones(int(window_len), "d")
    elif window == "sg_smooth":
        w = np.array([0.1, 0.25, 0.3, 0.25, 0.1])
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")

    return y[int(window_len / 2) : len(x) + int(window_len / 2)]


def correct_attenuation(r, refl, proc_dp_phase_shift, temp):
    """
    Calculate the attenuation from a polarimetric radar using Z-PHI method.

    References
    ----------
    Gu et al. Polarimetric Attenuation Correction in Heavy Rain at C Band,
    JAMC, 2011, 50, 39-58.

    """
    a_coef = 0.06
    beta = 0.8

    init_refl_correct = refl + proc_dp_phase_shift * a_coef
    dr = (r[1] - r[0]) / 1000.0

    specific_atten = np.zeros(refl.shape, dtype="float32")
    atten = np.zeros(refl.shape, dtype="float32")

    mask = refl.mask | (temp < 0)

    for i in range(refl.shape[0]):
        ray_phase_shift = proc_dp_phase_shift[i, :]
        ray_init_refl = init_refl_correct[i, :]

        last_six_good = np.where(~mask[i, :])[0][-6:]
        phidp_max = np.median(ray_phase_shift[last_six_good])
        sm_refl = smooth_and_trim(ray_init_refl, window_len=5)
        reflectivity_linear = 10.0 ** (0.1 * beta * sm_refl)
        reflectivity_linear[np.isnan(reflectivity_linear)] = 0
        self_cons_number = 10.0 ** (0.1 * beta * a_coef * phidp_max) - 1.0
        I_indef = cumulative_trapezoid(0.46 * beta * dr * reflectivity_linear[::-1])
        I_indef = cumulative_trapezoid(0.46 * beta * dr * reflectivity_linear[::-1])
        I_indef = np.append(I_indef, I_indef[-1])[::-1]

        # set the specific attenutation and attenuation
        specific_atten[i, :] = reflectivity_linear * self_cons_number / (I_indef[0] + self_cons_number * I_indef)

        atten[i, :-1] = cumulative_trapezoid(specific_atten[i, :]) * dr * 2.0
        atten[i, :-1] = cumulative_trapezoid(specific_atten[i, :]) * dr * 2.0
        atten[i, -1] = atten[i, -2]

    return atten

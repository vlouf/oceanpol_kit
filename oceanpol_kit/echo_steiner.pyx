import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt, pow, log10, fabs
from libc.stdlib cimport exit
from libc.string cimport strcmp

# Fonction pour calculer le rayon convectif
cdef void convective_radius(double ze_bkg, char* area_relation, double* conv_rad):
    if strcmp(area_relation, b"small") == 0:
        if ze_bkg < 30.0:
            conv_rad[0] = 1000.0
        elif ze_bkg < 35.0:
            conv_rad[0] = 2000.0
        elif ze_bkg < 40.0:
            conv_rad[0] = 3000.0
        elif ze_bkg < 45.0:
            conv_rad[0] = 4000.0
        else:
            conv_rad[0] = 5000.0
    elif strcmp(area_relation, b"medium") == 0:
        if ze_bkg < 25.0:
            conv_rad[0] = 1000.0
        elif ze_bkg < 30.0:
            conv_rad[0] = 2000.0
        elif ze_bkg < 35.0:
            conv_rad[0] = 3000.0
        elif ze_bkg < 40.0:
            conv_rad[0] = 4000.0
        else:
            conv_rad[0] = 5000.0
    elif strcmp(area_relation, b"large") == 0:
        if ze_bkg < 20.0:
            conv_rad[0] = 1000.0
        elif ze_bkg < 25.0:
            conv_rad[0] = 2000.0
        elif ze_bkg < 30.0:
            conv_rad[0] = 3000.0
        elif ze_bkg < 35.0:
            conv_rad[0] = 4000.0
        else:
            conv_rad[0] = 5000.0
    elif strcmp(area_relation, b"sgp") == 0:
        if ze_bkg < 40.0:
            conv_rad[0] = 0.0
        elif ze_bkg < 45.0:
            conv_rad[0] = 1000.0
        elif ze_bkg < 50.0:
            conv_rad[0] = 2000.0
        elif ze_bkg < 55.0:
            conv_rad[0] = 6000.0
        else:
            conv_rad[0] = 8000.0
    else:
        exit(1)

        
# Fonction pour calculer la "peakedness"
cdef void peakedness(double ze_bkg, char* peak_relation, double* peak):
    if strcmp(peak_relation, b"default") == 0:
        if ze_bkg < 42.43:
            peak[0] = 10.0 - pow(ze_bkg, 2) / 180.0
        else:
            peak[0] = 0.0
    elif strcmp(peak_relation, b"sgp") == 0:
        if ze_bkg < 0.0:
            peak[0] = 14.0 - pow(ze_bkg, 2) / 180.0
        else:
            peak[0] = 4.0
    else:
        exit(1)


@cython.boundscheck(False)
@cython.wraparound(False)
def classify(
    np.ndarray[np.float64_t, ndim=3] ze,
    np.ndarray[np.float64_t, ndim=1] x,
    np.ndarray[np.float64_t, ndim=1] y,
    np.ndarray[np.float64_t, ndim=1] z,
    double dx, double dy, double intense, double bkg_rad, double work_level,
    str area_relation, str peak_relation, bint use_intense,
    double fill_value
):
    cdef int nx = x.shape[0]
    cdef int ny = y.shape[0]
    cdef int nz = z.shape[0]
    cdef int i, j, l, m, imin, imax, jmin, jmax, lmin, lmax, mmin, mmax, k
    cdef double conv_rad, peak, ze_bkg, rad, n, sum_ze, min_diff, diff
    cdef np.ndarray[np.int32_t, ndim=2] sclass = np.zeros((ny, nx), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] ze_s = np.zeros((ny, nx), dtype=np.float64)
    cdef np.ndarray[np.uint8_t, ndim=2] m_ze = np.zeros((ny, nx), dtype=np.uint8)
    cdef bytes area_relation_c = area_relation.encode('utf-8')
    cdef bytes peak_relation_c = peak_relation.encode('utf-8')
    
    # Extract constant-height cross-section of reflectivity
    k = np.argmin(np.abs(z - work_level))
    for j in range(ny):
        for i in range(nx):
            ze_s[j, i] = ze[k, j, i]
            m_ze[j, i] = ze_s[j, i] != fill_value
    
    # Classification algorithm
    for i in range(nx):
        imin = max(0, i - int(bkg_rad / dx))
        imax = min(nx - 1, i + int(bkg_rad / dx))
        
        for j in range(ny):
            if m_ze[j, i]:
                if sclass[j, i] == 0:
                    jmin = max(0, j - int(bkg_rad / dy))
                    jmax = min(ny - 1, j + int(bkg_rad / dy))
                    
                    n = 0.0
                    sum_ze = 0.0
                    
                    for l in range(imin, imax + 1):
                        for m in range(jmin, jmax + 1):
                            if m_ze[m, l]:
                                rad = sqrt(pow(x[l] - x[i], 2) + pow(y[m] - y[j], 2))
                                if rad <= bkg_rad:
                                    n += 1.0
                                    sum_ze += pow(10.0, ze_s[m, l] / 10.0)
                                    
                    ze_bkg = 10.0 * log10(sum_ze / n)                    
                    # Calculate convective radius
                    convective_radius(ze_bkg, area_relation_c, &conv_rad)
                    
                    lmin = max(0, i - int(conv_rad / dx))
                    lmax = min(nx - 1, i + int(conv_rad / dx))
                    mmin = max(0, j - int(conv_rad / dy))
                    mmax = min(ny - 1, j + int(conv_rad / dy))
                    
                    if use_intense and ze_s[j, i] >= intense:
                        sclass[j, i] = 2
                        for l in range(lmin, lmax + 1):
                            for m in range(mmin, mmax + 1):
                                if m_ze[m, l]:
                                    rad = sqrt(pow(x[l] - x[i], 2) + pow(y[m] - y[j], 2))
                                    if rad <= conv_rad:
                                        sclass[m, l] = 2
                    else:
                        peakedness(ze_bkg, peak_relation_c, &peak)
                        if ze_s[j, i] - ze_bkg >= peak:
                            sclass[j, i] = 2
                            for l in range(imin, imax + 1):
                                for m in range(jmin, jmax + 1):
                                    if m_ze[m, l]:
                                        rad = sqrt(pow(x[l] - x[i], 2) + pow(y[m] - y[j], 2))
                                        if rad <= conv_rad:
                                            sclass[m, l] = 2
                        else:
                            sclass[j, i] = 1
    return sclass

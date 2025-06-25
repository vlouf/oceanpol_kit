#include <stdio.h>
#include <math.h> 
#include <stdlib.h>
#include <stdbool.h>
#include <string.h> 

void convective_radius(double ze_bkg, const char* area_relation, double* conv_rad) {
    // Determine convective radius based on area_relation and ze_bkg
    if (strcmp(area_relation, "small") == 0) {
        if (ze_bkg < 30.0) {
            *conv_rad = 1000.0;
        } else if (ze_bkg >= 30.0 && ze_bkg < 35.0) {
            *conv_rad = 2000.0;
        } else if (ze_bkg >= 35.0 && ze_bkg < 40.0) {
            *conv_rad = 3000.0;
        } else if (ze_bkg >= 40.0 && ze_bkg < 45.0) {
            *conv_rad = 4000.0;
        } else {
            *conv_rad = 5000.0;
        }
    } else if (strcmp(area_relation, "medium") == 0) {
        if (ze_bkg < 25.0) {
            *conv_rad = 1000.0;
        } else if (ze_bkg >= 25.0 && ze_bkg < 30.0) {
            *conv_rad = 2000.0;
        } else if (ze_bkg >= 30.0 && ze_bkg < 35.0) {
            *conv_rad = 3000.0;
        } else if (ze_bkg >= 35.0 && ze_bkg < 40.0) {
            *conv_rad = 4000.0;
        } else {
            *conv_rad = 5000.0;
        }
    } else if (strcmp(area_relation, "large") == 0) {
        if (ze_bkg < 20.0) {
            *conv_rad = 1000.0;
        } else if (ze_bkg >= 20.0 && ze_bkg < 25.0) {
            *conv_rad = 2000.0;
        } else if (ze_bkg >= 25.0 && ze_bkg < 30.0) {
            *conv_rad = 3000.0;
        } else if (ze_bkg >= 30.0 && ze_bkg < 35.0) {
            *conv_rad = 4000.0;
        } else {
            *conv_rad = 5000.0;
        }
    } else if (strcmp(area_relation, "sgp") == 0) {
        if (ze_bkg < 40.0) {
            *conv_rad = 0.0;
        } else if (ze_bkg >= 40.0 && ze_bkg < 45.0) {
            *conv_rad = 1000.0;
        } else if (ze_bkg >= 45.0 && ze_bkg < 50.0) {
            *conv_rad = 2000.0;
        } else if (ze_bkg >= 50.0 && ze_bkg < 55.0) {
            *conv_rad = 6000.0;
        } else {
            *conv_rad = 8000.0;
        }
    } else {
        fprintf(stderr, "Invalid area_relation: %s\n", area_relation);
        exit(1); // Stop execution for invalid input
    }
}

void peakedness(double ze_bkg, const char* peak_relation, double* peak) {
    // Determine the necessary peakedness based on peak_relation and ze_bkg
    if (strcmp(peak_relation, "default") == 0) {
        if (ze_bkg < 0.0) {
            *peak = 10.0;
        } else if (ze_bkg >= 0.0 && ze_bkg < 42.43) {
            *peak = 10.0 - pow(ze_bkg, 2) / 180.0;
        } else {
            *peak = 0.0;
        }
    } else if (strcmp(peak_relation, "sgp") == 0) {
        if (ze_bkg < 0.0) {
            *peak = 14.0;
        } else if (ze_bkg >= 0.0 && ze_bkg < 42.43) {
            *peak = 14.0 - pow(ze_bkg, 2) / 180.0;
        } else {
            *peak = 4.0;
        }
    } else {
        fprintf(stderr, "Invalid peak_relation: %s\n", peak_relation);
        exit(1); // Stop execution for invalid input
    }
}

// Main classify function
void classify(
    const double* ze, const double* x, const double* y, const double* z,
    double dx, double dy, double intense, double bkg_rad, double work_level,
    const char* area_relation, const char* peak_relation, bool use_intense,
    double fill_value, int nx, int ny, int nz, int* sclass
) {
    // Local variables
    double conv_rad, peak, ze_bkg, rad, n, sum_ze;
    double ze_s[ny * nx];
    bool m_ze[ny * nx];
    int k, i, j, l, m, imin, imax, jmin, jmax, lmin, lmax, mmin, mmax;

    // Find the index of the working level
    k = 0;
    double min_diff = fabs(z[0] - work_level);
    for (int level = 1; level < nz; level++) {
        double diff = fabs(z[level] - work_level);
        if (diff < min_diff) {
            k = level;
            min_diff = diff;
        }
    }

    // Extract constant-height cross-section of reflectivity
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            ze_s[j * nx + i] = ze[k * ny * nx + j * nx + i];
            m_ze[j * nx + i] = ze_s[j * nx + i] != fill_value;
        }
    }

    // Initialize sclass array with zeros (undefined)
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            sclass[j * nx + i] = 0;
        }
    }

    // Classification algorithm
    for (i = 0; i < nx; i++) {
        imin = (int)fmax(0, i - bkg_rad / dx);
        imax = (int)fmin(nx - 1, i + bkg_rad / dx);

        for (j = 0; j < ny; j++) {
            if (m_ze[j * nx + i]) {
                if (sclass[j * nx + i] == 0) {
                    jmin = (int)fmax(0, j - bkg_rad / dy);
                    jmax = (int)fmin(ny - 1, j + bkg_rad / dy);

                    n = 0.0;
                    sum_ze = 0.0;

                    for (l = imin; l <= imax; l++) {
                        for (m = jmin; m <= jmax; m++) {
                            if (m_ze[m * nx + l]) {
                                rad = sqrt(pow(x[l] - x[i], 2) + pow(y[m] - y[j], 2));
                                if (rad <= bkg_rad) {
                                    n += 1.0;
                                    sum_ze += pow(10.0, ze_s[m * nx + l] / 10.0);
                                }
                            }
                        }
                    }

                    ze_bkg = 10.0 * log10(sum_ze / n);

                    // Calculate convective radius
                    convective_radius(ze_bkg, area_relation, &conv_rad);

                    lmin = (int)fmax(0, i - conv_rad / dx);
                    lmax = (int)fmin(nx - 1, i + conv_rad / dx);
                    mmin = (int)fmax(0, j - conv_rad / dy);
                    mmax = (int)fmin(ny - 1, j + conv_rad / dy);

                    if (use_intense && ze_s[j * nx + i] >= intense) {
                        sclass[j * nx + i] = 2;

                        for (l = lmin; l <= lmax; l++) {
                            for (m = mmin; m <= mmax; m++) {
                                if (m_ze[m * nx + l]) {
                                    rad = sqrt(pow(x[l] - x[i], 2) + pow(y[m] - y[j], 2));
                                    if (rad <= conv_rad) {
                                        sclass[m * nx + l] = 2;
                                    }
                                }
                            }
                        }
                    } else {
                        peakedness(ze_bkg, peak_relation, &peak);

                        if (ze_s[j * nx + i] - ze_bkg >= peak) {
                            sclass[j * nx + i] = 2;

                            for (l = imin; l <= imax; l++) {
                                for (m = jmin; m <= jmax; m++) {
                                    if (m_ze[m * nx + l]) {
                                        rad = sqrt(pow(x[l] - x[i], 2) + pow(y[m] - y[j], 2));
                                        if (rad <= conv_rad) {
                                            sclass[m * nx + l] = 2;
                                        }
                                    }
                                }
                            }
                        } else {
                            sclass[j * nx + i] = 1; // Stratiform
                        }
                    }
                }
            }
        }
    }
}

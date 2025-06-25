#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

void convective_radius(double ze_bkg, const char* area_relation, double* conv_rad);
void peakedness(double ze_bkg, const char* peak_relation, double* peak);
void classify(
    const double* ze, const double* x, const double* y, const double* z,
    double dx, double dy, double intense, double bkg_rad, double work_level,
    const char* area_relation, const char* peak_relation, bool use_intense,
    double fill_value, int nx, int ny, int nz, int* sclass
);

#endif // CLASSIFICATION_H

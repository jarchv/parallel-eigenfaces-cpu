#ifndef image_hpp
#define image_hpp

#include "tools.hpp"

void    showImage(double *I, int w, int h);
void    readImages(double **X, int folders, int nimgs, int h, int w);
double* Reconstructor(double *Img, double **Wk, int m, int k);

#endif
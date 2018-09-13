#ifndef nn_hpp
#define nn_hpp

#include <stdlib.h>
#include "tools.hpp"

double **getW(int n, int m);
double *getBias(int n, double init);
double *getPred(double *input, int isize, int osize, double **nn_W, double *nn_b);
double *softMax(double *pred_in, int sizeo);
double crossEntropy(double *output, double *label, int sizeo);
void backProp(double **nnW,  double *nnb, double *input, int size_input, int sizeout, double *label, double *output, double learning_rate);

#endif



#ifndef tools_hpp
#define tools_hpp

#include <iostream>
#include <string>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <limits>
#include <omp.h>
#include <stdio.h>

#define PATH "faces/"

extern int thread_count;

using namespace std;
using namespace cv;	

double  norm(double *A, int n);
int     maxIndx(double *pred_in, int size);
void    VecNormalizer(double *A, int n);
void    printVec(double *v, int vec_size);
void    getS(double **S, double **X, double **Xm, double **Xm_t, double *V, int X_rows, int X_cols);
void    getW(double **A, double **B, double **W, int m, int n);


#endif
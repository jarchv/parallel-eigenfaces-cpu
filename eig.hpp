#ifndef eig_hpp
#define eig_hpp

#include "tools.hpp"

void    copyV(double*, double*, int);
void    copyM(double**, double**, int);
void    matmul(double**, double*, double*, int);
void    updateAA(double**, double, double*, int);
void    copy2Mat(double **V, double *Vi,int col, int n);
double  dotV(double *V1, double *V2, double n);
void    vectorOp(double *A, double *B, double *C,int n, int op);
double  normdiff(double *A, double *B, int n);
void    eigenfn(double **A, double **eigenVec, double *eigenVal, int n, double tol);
double* getProyection(double *Img, double **Wk, int m, int k);

#endif
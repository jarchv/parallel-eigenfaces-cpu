#include <iostream>
#include <stdio.h>
#include <string>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <omp.h>

#include "eig.hpp"
#include "images.hpp"
#include "nn.h"

#define PATH "faces/"

using namespace std;
using namespace cv;

void readImages(double **X, int folders, int nimgs, int rows, int cols);
void getS(double **S, double **X, double **Xm, double **Xm_t, double *V, int X_row, int X_cols);
void getVD(double **S, double *lambda, double **eigV, double tol1, double tol2);
void showImage(double **I, int j, int w, int h);

int main(int argc, char *argv[]){
    
    int             i,j,k;
    int             nfiles = 40;        //  # folders
    int             nimages  = 10;      //  # imgs x folder

//  Matrix X, Xm, Xm_t
    double          **X;
    double          **Xm;
    double          **Xm_t;
    double          **S;
    double          **eigenVec;
    double          *eigenVal;
    double          *muV;
    int             X_rows=nfiles*nimages;
    int             imgw  = 92;
    int             imgh  = 112;
    int             X_cols= imgh*imgw;   //  h = 112, w = 92

    X           = new double*[X_rows];
    Xm          = new double*[X_rows];
    Xm_t        = new double*[X_cols];
    S           = new double*[X_rows];
    muV         = new double[X_cols];    
    
    for(int ni=0; ni<X_rows; ni++){
        X[ni]   = new double[X_cols];
        Xm[ni]  = new double[X_cols];
        S[ni]   = new double[X_rows];
    }

    for(int pi=0; pi<X_cols; pi++){
        Xm_t[pi] = new double[X_rows];
    }

    double **W = new double*[imgh*imgw];
    for(int iw=0; iw<imgh*imgw; iw++){
        W[iw] = new double[X_rows];
    }   

    double *D = new double[X_rows];
    double **V = new double*[X_rows];
    for(int i=0; i<X_rows; i++){
        V[i] = new double[X_rows];
    }

    double tol = 1.0e-20;
    int pdim = imgh * imgw; 
    readImages(X,nfiles,nimages,imgh,imgw);
    getS(S,X,Xm,Xm_t, muV, X_rows, X_cols);
    eigenfn(S,V,D,X_rows,tol);
    getW(Xm_t, V, W, pdim, X_rows);
    
    double *temp = new double[pdim];

    for(int j=0; j < 16; j++){
        for(int i=0; i< pdim; i++){
            temp[i] = W[i][j];
        }

        showImage(temp, imgw, imgh);
    }
    
    for(int ir=10; ir < 400; ir+=10){
        temp = Reconstructor(Xm[0], W, pdim, ir);
        cout<<"\tk -> "<<ir<<endl;
        showImage(temp, imgw, imgh);   
    }

    double **W      = getW(40, 300, 0.01);
    double  *b      = getBias(40, 0.0);
    double *pred     = new double[40];
    
    double loss;
    double label[40];

    for (int lb=0; lb < 40; lb++){
        label[lb] = 0.0;
    }
    
    for (int epoch=0; epoch < 100; epoch++){
        loss = 0.0;
        for (int i = 0; i < X_rows; i++){
            pred = getPred(Im[i], 40, X_rows, W, b);
            softMax(pred, 40);
            loss += crossEntropy(double *output, double *label, int sizeo)
            backProp(W,Im[i],pdim,40,b,i)
        }
        
        loss /= X_rows;
        cout << "loss \t:\t" << loss << endl;         
    }


    getPred(input, int isize, int osize, double **W, double *b)
     
//  free
    for(int ni=0; ni<X_rows; ni++){
        delete [] X[ni];
        delete [] Xm[ni];
    }
    
    for(int pi=0; pi<X_cols; pi++){
        delete [] Xm_t[pi];    
    }
    delete [] X;
    delete [] Xm;
    delete [] Xm_t;
    delete [] temp;

    return 0;
}

void readImages(double **X, int folders, int nimgs, int h, int w){
    string      ifolder;
    string      jimage;
    string      filename;

    int         n=0;

    for(int i=1; i<=folders; i++){
        for(int j=1; j<=nimgs; j++){
            ifolder     = "s" + to_string(i)+"/";
            jimage      = to_string(j) + ".pgm";
            filename    = PATH + ifolder + jimage;
//            cout<<filename<<endl;
            Mat I = imread(filename, IMREAD_GRAYSCALE);
//          Get X
            n = (i-1)*nimgs + (j-1);
        
            for(int ri=0; ri<h; ri++){
                for(int ci=0; ci<w; ci++){ 
                    X[n][ri*w + ci] = (double)I.at<uint8_t>(ri,ci);
                }
            }             
        }
    }
    cout <<"\nImages:\n"<<endl;
    cout <<"\t#elements(X.rows) = "<<n+1<<endl;  
}

void getS(double **S, double **X, double **Xm, double **Xm_t, double *V, int X_rows, int X_cols){
  //   Get Xm (all column with mu_j = 0)
    int mu;
    for(int pi=0; pi<X_cols; pi++){
        mu = 0;
        for(int ni=0; ni<X_rows; ni++){
            mu += X[ni][pi];
        }
        mu = mu/((double)X_rows);
        V[pi] = mu;
        for(int ni=0; ni<X_rows; ni++){
            Xm[ni][pi] = X[ni][pi] - mu;
        }
    }
    cout<<"\nXm \t\t<-"<<endl;

//  Get transpose
    for(int ni=0; ni<X_rows; ni++){
        for(int pi=0; pi<X_cols; pi++){
            Xm_t[pi][ni] = Xm[ni][pi];
        }
    }
    cout<<"Xm_t \t\t<-"<<endl;    

//  Get S
    double temp;
    for(int sr=0; sr<X_rows; sr++){
        for(int sc=0; sc<X_rows; sc++){
            temp=0.0;
            for(int it=0; it<X_cols; it++){
                temp += Xm[sr][it]*Xm_t[it][sc]; 
            }
            S[sr][sc] = temp;
        }
    }    
    cout<<"S = Xm*Xm_t \t<-"<<endl;   
}

/**
*   @autor      : José Chávez    
*   @file       : main.cpp
*   @viersion   : v1.0
*   @purpose    : Parallel implementation of eigenfaces
*/

#include <iostream>
#include <stdio.h>
#include <string>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <omp.h>

#include "tools.hpp"
#include "eig.hpp"
#include "images.hpp"
#include "nn.hpp"

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
    int             file_imgs  = 10;      //  # imgs x folder

//  Matrix X, Xm, Xm_t
    double          **X;
    double          **Xm;
    double          **Xm_t;
    double          **S;
    double          **eigenVec;
    double          *eigenVal;
    double          *muV;
    int             X_rows=nfiles*file_imgs;
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
    readImages(X,nfiles,file_imgs,imgh,imgw);
    getS(S,X,Xm,Xm_t, muV, X_rows, X_cols);
    eigenfn(S,V,D,X_rows,tol);
    getW(Xm_t, V, W, pdim, X_rows);
    
    double *temp = new double[pdim];

    //for(int j=0; j < 16; j++){
    //    for(int i=0; i< pdim; i++){
    //        temp[i] = W[i][j];
    //    }

    //    showImage(temp, imgw, imgh);
    //}

    //for(int ir=10; ir < X_rows; ir+=10){
    //    temp = Reconstructor(Xm[0], W, pdim, ir);
    //    cout<<"\tk -> "<<ir<<endl;
    //    showImage(temp, imgw, imgh);   
    //}

    //int key = 300;
    //cout<<"set key = "<<key<<endl;
    //double **Y = new double*[X_rows];
    //for (int i = 0; i < X_rows; i++)
    //{
    //    Y[i] = Reconstructor[Xm[i], W, pdim, key]
    //}
    double **nn_W      = getW(nfiles, pdim);
    double  *nn_b      = getBias(nfiles, 0.0);
    double *pred       = new double[nfiles];
    
    double loss;
    double **label     = new double*[X_rows];
    double *output;

    for (int in=0; in < X_rows; in++){
        label[in] = new double[nfiles];
    }
    
    for (int in=0; in < X_rows; in++){
        for (int il = 0; il < nfiles; il++)
        {
            label[in][il] = (double)((in / 10) == il); 
        }
    } 
    double lr = 0.01;

    for (int epoch=0; epoch < 2; epoch++){
        loss = 0.0;
        for (int i = 0; i < X_rows; i++){
            pred = getPred(Xm[i],pdim, nfiles, nn_W, nn_b);
            cout<<"\t pred("<<i<<")="<<output<<endl;
            output = softMax(pred, nfiles);
            cout<<"\t output("<<i<<")="<<output<<endl;
            loss += crossEntropy(output,label[i], nfiles);
            cout<<"\t loss("<<i<<")="<<loss<<endl;
            backProp(nn_W,nn_b,Xm[i],X_rows,nfiles,label[i], output, lr);
        }
        
        loss /= (double)X_rows;
        cout << "loss epoch\t:\t" << loss << endl;         
    }
     
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
    cout <<"#elements(X.rows) = "<<n+1<<endl;  
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
    cout<<"\n->\tXm"<<endl;

//  Get transpose
    for(int ni=0; ni<X_rows; ni++){
        for(int pi=0; pi<X_cols; pi++){
            Xm_t[pi][ni] = Xm[ni][pi];
        }
    }
    cout<<"->\tXm_t"<<endl;    

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
    cout<<"-> \tS = Xm*Xm_t"<<endl;   
}

#include <iostream>
#include <stdio.h>
#include <string>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include "tools.hpp"

#define PATH "faces/"
using namespace std;
using namespace cv;

void readImages(double **X, int folders, int nimgs, int rows, int cols);
void getS(double **S, double **X, double **Xm, double **Xm_t, double *V, int X_row, int X_cols);
void getVD(double **S, double *lambda, double **eigV, double tol1, double tol2);

int main(int argc, char *argv[]){
    int         i,j,k;
    int         nfiles = 40;      //  # folders
    int         nimages  = 10;      //  # imgs x folder



//  Matrix X, Xm, Xm_t
    double       **X;
    double       **Xm;
    double       **Xm_t;
    double       **S;
    double       **eigenVec;
    double       *eigenVal;
    double       *muV;
    int         X_rows=nfiles*nimages;
    int         imgw  = 92;
    int         imgh  = 112;
    int         X_cols= imgh*imgw;   //  h = 112, w = 92

    X       = new double*[X_rows];
    Xm      = new double*[X_rows];
    Xm_t    = new double*[X_cols];
    S       = new double*[X_rows];

    muV     = new double[X_cols];    
    
    for(int ni=0; ni<X_rows; ni++){
        X[ni]   = new double[X_cols];
        Xm[ni]  = new double[X_cols];
        S[ni]   = new double[X_rows];
    }

    for(int pi=0; pi<X_cols; pi++){
        Xm_t[pi] = new double[X_rows];
    }

    
    readImages(X,nfiles,nimages,imgh,imgw);
    getS(S,X,Xm,Xm_t, muV, X_rows, X_cols);


    double *D = new double[X_rows];
    double **V = new double*[X_rows];
    for(int i=0; i<X_rows; i++){
        V[i] = new double[X_rows];
    }
    double tol = 1.0e-20;
    eigenfn(S,V,D,X_rows,tol);
    
    int pdim = imgh*imgw;
    double **W = new double*[pdim];
    for(int iw=0; iw<pdim; iw++){
        W[iw] = new double[X_rows];
    }
    
    getW(Xm_t, V, W, pdim, X_rows);
    

    //for(int ivec=0; ivec<)
    double_t *matting = new double_t[pdim];
    double *mm = new double[pdim];
    for (int i=0; i<pdim; i++){
        matting[i] = (double_t)W[i][0];
        mm[i] = W[i][0];
        cout << "W["<<i<<"][0]="<<W[i][0]<<endl;     
    }
    cout<<"lambda: "<<D[0]<<endl;
    
    cout<<"norm: "<<norm(mm,pdim)<<endl;
    Mat sample(imgh,imgw,CV_64FC1, matting);

    Mat dst;
    normalize(sample, dst, 0.0,1.0, NORM_MINMAX, CV_64FC1);
    imshow("image", dst);
    waitKey(0);
        
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

    return 0;
}

void readImages(double **X, int folders, int nimgs, int h, int w){
//    namedWindow("image",WINDOW_AUTOSIZE);
    string      ifolder;
    string      jimage;
    string      filename;

    int         n=0;

    for(int i=1; i<=folders; i++){
        for(int j=1; j<=nimgs; j++){
            ifolder     = "s" + to_string(i)+"/";
            jimage      = to_string(j) + ".pgm";
            filename    = PATH + ifolder + jimage;
            cout<<filename<<endl;
            Mat I = imread(filename, IMREAD_GRAYSCALE);

//            display image
//            imshow("image", I);
//            waitKey(0);
           
            //I.convertTo(I, CV_64F);
            //double_t *var = new double_t[h*w];
//          Get X
            n = (i-1)*nimgs + (j-1);
        
            for(int ri=0; ri<h; ri++){
                for(int ci=0; ci<w; ci++){ 
                    X[n][ri*w + ci] = (double)I.at<uint8_t>(ri,ci);
                }
            }
             
            //for(int i = 0; i<h*w; i++){
            //    var[i] = (double_t)(X[n][i]/255 - 0.5);
            //}
            //Mat sam(h,w,CV_64FC1, var);
            //Mat dst;
            //normalize(sam, dst, 0.0, 1.0, NORM_MINMAX, CV_64FC1);
            //sam.convertTo(dst, CV_8UC1);
            //imshow("dst",dst);
            //imshow("sam",sam);
            //imshow("I",I);
            //waitKey(0);                     
        }
    }
    cout <<"\nreadImages:\n"<<endl;
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
    cout<<"Xm \t\t<-"<<endl;

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


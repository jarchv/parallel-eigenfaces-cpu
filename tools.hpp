#include <iostream>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <limits>
#include <omp.h>

#define PATH "faces/"

int thread_count;

using namespace std;
using namespace cv;	

int maxIndx(double *pred_in, int size){
	double temp = pred_in[0];
	int indx = 0;
	for (int i = 1; i < size; i++)
	{
		if (pred_in[i] > temp){
			indx = i;
			temp = pred_in[i];
		}
	}

	return indx;
}

double norm(double *A, int n){
    double temp=0.0;

    for(int i=0; i<n; i++){
        temp+= A[i]*A[i];
    }
    return sqrt(temp);
}

void VecNormalizer(double *A, int n){
    double mod = norm(A,n);
    for(int i=0; i<n; i++){
        A[i]/=mod;
    }
}

void printVec(double *v, int vec_size){
	cout<<"[";
	for (int i = 0; i < vec_size-1; i++)
	{
		cout<<v[i]<<",";
	}
	cout<<v[vec_size-1]<<"]"<<endl;
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

            Mat I = imread(filename, IMREAD_GRAYSCALE);

            n = (i-1)*nimgs + (j-1);

#           pragma omp parallel for num_threads(thread_count)
            for(int ri=0; ri<h; ri++){
                for(int ci=0; ci<w; ci++){ 
                    X[n][ri*w + ci] = (double)I.at<uint8_t>(ri,ci);
                }
            }             
        }
    }
    cout <<"\n#elements(X.rows) = "<<folders*nimgs<<endl;  
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
    cout<<"->\tXm_t\t= Xm.T"<<endl;    

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
    cout<<"-> \tS\t= Xm*Xm_t"<<endl;   
}
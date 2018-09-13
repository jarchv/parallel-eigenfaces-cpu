#include "tools.hpp"

int thread_count;

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

void getW(double **A, double **B, double **W, int m, int n){
    cout<<"\nGetting W ...\n"<<endl;
    double temp;

    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            temp = 0.0;
            for(int k=0; k<n; k++){
                temp += A[i][k]*B[k][j];                
            }
            W[i][j] = temp;
        }
    }
    
    double *vecW_j = new double[m];
    for(int jW=0; jW<n; jW++){
        for(int iW=0; iW<m; iW++){
            vecW_j[iW] = W[iW][jW];
        }
        VecNormalizer(vecW_j, m);
        for(int iW=0; iW<m; iW++){
            W[iW][jW] = vecW_j[iW];
        }       
    }
}

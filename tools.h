#include <math.h>
#include <cmath>
#include <stdio.h>

using namespace std;



void normalize(double *A, int n);
void copyV(double *v1, double *v2, int n){
    for(int i=0; i<n; i++){
        v2[i] = v1[i];
    }
}

void copyM(double **m1, double **m2, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            m2[i][j]=m1[i][j];
        }
    }
}

void matmul(double **X, double *B, double *C, int n){
    double temp;

    for(int iA=0; iA<n;iA++){        
        temp = 0.0;
        for(int k=0;k<n; k++){
            temp+=X[iA][k]*B[k];
        }
        C[iA]=temp;
    }
}

void updateAA(double **AA, double lambda, double *xnew, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            AA[i][j] -= lambda*xnew[i]*xnew[j];
        }
    }
}

void copy2Mat(double **V, double *Vi,int col, int n){
    for(int i=0; i<n; i++){
        V[i][col] = Vi[i];
    }
}


double dotV(double *V1, double *V2, double n){
    double temp = 0.0;
    for(int i=0; i<n; i++){
        temp+=V1[i]*V2[i];
    }
    return temp;
}

void vectorOp(double *A, double *B, double *C,int n, int op){
    for(int i=0; i<n; i++){
        C[i] = A[i] + op*B[i];
    }
}

double norm(double *A, int n){
    double temp=0.0;
    for(int i=0; i<n; i++){
        temp+= A[i]*A[i];
    }

    return sqrt(temp);
}

double normdiff(double *A, double *B, int n){
    double temp=0.0;
    for(int i=0; i<n; i++){
        temp+= pow(A[i]-B[i],2); 
    }
    return sqrt(temp);
}

void normalize(double *A, int n){
    double mod = norm(A,n);
    for(int i=0; i<n; i++){
        A[i]/=mod;
    }
}

void eigenfn(double **A, double **eigenVec, double *eigenVal, int n, double tol){
    cout<<"\neigenfn...\n"<<endl;
    double **AA = new double*[n];
    for(int i=0; i<n; i++){
        AA[i] = new double[n];
    }

    copyM(A,AA,n);
    double *xini = new double[n];
    for(int j=0;j<n;j++){
        if(j==0){xini[j] = 1.0;}
        else{xini[j] = 0.0;}
    }
    double *x0 = new double[n];
    double *xnew = new double[n];
    double lambda; 
    double ol_lambda;
    
    for(int iv=0; iv<n;iv++){
        printf("\r%3d%%",(int)(100*(iv+1)/n));
        fflush(stdout);        
        copyV(xini,x0,n);
        ol_lambda = 0.0;
        int it = 0;
        while(1){
            matmul(AA,x0,xnew,n);
            lambda = norm(xnew,n);
            normalize(xnew,n);
            
            if((abs((ol_lambda-lambda)/ol_lambda)<tol) and (normdiff(x0,xnew,n)<tol)){
                break;
            }

            ol_lambda = lambda;
            copyV(xnew,x0,n); 

            if(it > 1000){break;}
            it++;           
        }
        double fac = dotV(xnew,xini,n);
        vectorOp(xini,xnew,xini,n,-fac);
        eigenVal[iv] = lambda;
        copy2Mat(eigenVec,xnew,iv,n);
        updateAA(AA,lambda,xnew,n);
    }
    cout<<endl;
    cout<<"\nDone!"<<endl;
}

void getW(double **A, double **B, double **W, int m, int n){
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
        normalize(vecW_j, m);
        for(int iW=0; iW<m; iW++){
            W[iW][jW] = vecW_j[iW];
        }       
    }
}

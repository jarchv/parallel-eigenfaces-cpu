double norm(double *A, int n){
    double temp=0.0;
    for(int i=0; i<n; i++){
        temp+= A[i]*A[i];
    }
    return sqrt(temp);
}

void normalize(double *A, int n){
    double mod = norm(A,n);
    for(int i=0; i<n; i++){
        A[i]/=mod;
    }
}

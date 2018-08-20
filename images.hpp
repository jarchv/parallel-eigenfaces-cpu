#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


void getW(double **A, double **B, double **W, int m, int n){
    cout<<"\nGetting W ...\n"<<endl;
    double temp;

#   pragma omp parallel for num_threads(thread_count)
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

void showImage(double *I, int w, int h){
    double_t *temp = new double_t[w*h];    
    
    for(int i=0; i < w*h; i++){
        temp[i] = (double_t)I[i];
    }

    Mat src(h,w,CV_64FC1,temp);
    Mat dst;
    normalize(src,dst,0.0,1.0,NORM_MINMAX,CV_64FC1);
    imshow("img", dst);
    waitKey(0);

    delete [] temp;
}

double *Reconstructor(double *Img, double **Wk, int m, int k){
    double *Y       = new double[k];
    double *newI    = new double[m];
    double temp;

    for(int jk=0; jk < k; jk++){
        temp = 0.0;
        for(int im=0; im<m; im++){
            temp += Img[im]*Wk[im][jk];
        }
        Y[jk] = temp;
    }
    
    for(int im=0; im < m; im++){
        temp =0.0;
        for(int ik=0; ik<k; ik++){
            temp+= Y[ik]*Wk[im][ik];
        }
        newI[im] = temp;
    }
    return newI;
}
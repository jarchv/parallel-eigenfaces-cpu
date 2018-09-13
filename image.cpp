#include "image.hpp"
#include "tools.hpp"

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
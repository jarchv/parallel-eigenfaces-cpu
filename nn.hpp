#include <math.h>
#include <stdlib.h>

double **getW(int n, int m){
	double **wtemp = new double*[n];
	for(int in=0; in < n; in++){
		wtemp[in] = new double[m];
	}
	for(int in=0; in < n; in++){
		for(int im=0; im < m; im++){
			wtemp[in][im] = (double)(rand()%100)/1000;
		}
	}
	return wtemp;
}

double *getBias(int n, double init){
	double *btemp = new double[n];
	for (int in = 0; in < n; ++in)
	{
		btemp[in] = init;
	}
	return btemp;
}

double *getPred(double *input, int isize, int osize, double **nn_W, double *nn_b){
	double temp;
	double *pred = new double[osize];

	for(int i=0; i < osize; i++){
		temp = 0.0;
		for(int j=0; j < isize; j++){
			temp += nn_W[i][j]*input[j];
		}

		temp += nn_b[i];
		pred[i] = temp;
	}

	return pred;
}

double *softMax(double *pred_in, int sizeo){
	double *output = new double[sizeo];
	double sum_temp= 0.0;
	
	for (int is = 0; is < sizeo; is++){
		output[is] = exp(pred_in[is]);
		sum_temp  += output[is];
	}
	
	for (int is = 0; is < sizeo; is++){
		output[is] = output[is]/sum_temp;
		//cout<<"softMax("<<is<<")="<<output[is]<<endl;;
	}

	return output;
}

double crossEntropy(double *output, double *label, int sizeo){
	double temp = 0.0;
	
	for (int is = 0; is < sizeo; is++){
		temp -= label[is]*log(output[is]); 
	}

	return temp;
}

void backProp(double **nnW,  double *nnb, double *input, int size_input, int sizeout, double *label, double *output, double learning_rate){
	double temp;

	for (int j=0; j < sizeout; j++){
		temp  	= -label[j]*(1-output[j]); //(-label[j]/output[j])*(output[j]*(1-output[j]));
		//temp   *= output[j]*(1-output[j]); 		
		//cout<<"temp: "<<temp<<endl;
		for (int i=0; i < size_input; i++){
			nnW[j][i] -= temp*input[i]*learning_rate;
			//cout<<"\tnnW["<<j<<"]["<<i<<"] ="<< nnW[j][i]<<endl;
		}
		nnb[j]   -= temp*learning_rate; 
	}
}





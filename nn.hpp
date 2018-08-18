#include <std.io>
#include <math.h>

double **getW(int n, int m, double init){
	double **wtemp = new double*[n];
	for(int in=0; in < n; in++){
		wtemp[in] = new double[m];
	}

	for(int in=0; in < n; in++){
		for(int im=0; im < n; im++){
			wtemp[in][im] = init;
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

double *getPred(double *input, int isize, int osize, double **W, double *b){
	double temp;
	double *pred = new double[osize];

	for(int i=0; i < osize; i++){
		temp = 0.0;
		for(int j=0; j < isize; j++){
			temp += W[i][j]*input[j]; 
		}

		temp += b[i];
		pred[i] = temp;
	}

	return pred;
}

double *softMax(double *pred_in, int sizeo){
	double *output = new double[sizeo];
	double sum_temp= 0.0;
	
	for (int is = 0; is < sizeo; is++){
		output[is] = exp(pred[is]);
		sum_temp  += output[is];
	}
	
	for (int is = 0; is < sizeo; is++){
		output[is] = output[is]/sum_temp;
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

void backProp(double **W, double *input, int sizeImg, int sizeout, double *b, double *label, double *output){
	double temp;

	for (int j=0; j < sizeout; j++){
		temp  	= -label[j]/output[j];
		temp   *= output[j]*(1-output[j]); 		
		
		for (int i=0; i < sizeImg; i++){
			W[j][i] -= temp*input[i];; 
		}
		b[j]   -= temp; 
	}
}





#include <stdlib.h>
#define thread_count 8

double **getW(int n, int m){
	double **wtemp = new double*[n];
	for(int in=0; in < n; in++){
		wtemp[in] = new double[m];
	}
	for(int in=0; in < n; in++){
		for(int im=0; im < m; im++){
			wtemp[in][im] = (double)(rand()%100);
		}
		VecNormalizer(wtemp[in], m);
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

#   pragma omp parallel num_threads(thread_count)
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
	double *output_temp = new double[sizeo];
	double sum_temp= 0.0;
	
	for (int is = 0; is < sizeo; is++){
		output_temp[is] = exp(pred_in[is]);
		sum_temp  += output_temp[is];
	}

	for (int is = 0; is < sizeo; is++){
		output_temp[is] = output_temp[is]/sum_temp;
	}

	return output_temp;
}

double crossEntropy(double *output, double *label, int sizeo){
	double temp = 0.0;
	
	for (int is = 0; is < sizeo; is++){
		temp -= label[is]*log(output[is]); 
	}

	return temp/sizeo;
}

void backProp(double **nnW,  double *nnb, double *input, int size_input, int sizeout, double *label, double *output, double learning_rate){
	double temp;

	for (int j=0; j < sizeout; j++){
		temp  	= -label[j]*(1-output[j]); //(-label[j]/output[j])*(output[j]*(1-output[j]));
		//temp   *= output[j]*(1-output[j]); 		
		//cout<<"temp: "<<temp<<endl;
#   	pragma omp parallel num_threads(thread_count)
		for (int i=0; i < size_input; i++){
			nnW[j][i] -= temp*input[i]*learning_rate;
			//cout<<"\tnnW["<<j<<"]["<<i<<"] ="<< nnW[j][i]<<endl;
		}
		nnb[j]   -= temp*learning_rate; 
	}
}





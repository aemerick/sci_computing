//===================================
// Author: Andrew Emerick
// Homework 1
// 10/11/13
// Serial Conjugate Gradient Solver
//===================================
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;
// (sorry for not using a header file)

template <class T> T *Create1D(int N1)
{
    T * array = new T  [N1];
    array = new T [N1];
    return array;
};

template <class T> T **Create2D(int N1, int N2)
{
    T ** array = new T * [N1];
    array = new T * [N1];
    array[0] = new T [N1*N2];

    for(int i = 0; i < N1; i++) {
        if (i < N1 ) {
            array[i+1] = &(array[0][(i+1)*N2]);
        }
   }
    return array;
};

template <class T> void Delete2D(T **array) {
    delete[] array;
};

template <class T> void Delete1D(T *array) {
    delete[] array;
};

template <class T>
void printArray(T* a, int N){ //prints a vector
//	int N = sizeof(a)/sizeof(a[0]);
	cout<<"[";
	for(int i=0;i<N;i++){
	
	if(i==0){cout<<setw(2)<<a[i];}
	else{cout<<","<<setw(2)<<a[i];}	
	}
	cout<<"]"<<endl;
}


//==============================================
// Solves the equation A x = b for unknown x using 
// the Conjugate Gradient algortithm assuming
// positive definite square matrix A
//===============================================
void CG(float** A, double* x, double* b, int N, double tol, double & err){
	int iter=0, max_iter=5*N;
	double *Ap_vec, *p, *r;
	double alpha, beta, sum1=0, sum2=0, sum3=0;
	
	p = Create1D<double> (N);
	r = Create1D<double >(N);
	Ap_vec = Create1D<double >(N);

      	err=100;
	for(int i=0; i<N; i++){ // set the initial conditions for the solver
		x[i] = 0;
		p[i] = b[i];
		r[i] = b[i];	
	}

	while(abs(err)>tol && iter<max_iter){ //loop until done (error is small)
		for(int i=0; i<N; i++){//set vector to zero
			Ap_vec[i] = 0; // vector will contain result from A * p
		}
	
		sum1=0; sum2=0; 
		//makes needed computations for alpha
		for(int i=0; i<N; i++){
			sum3=0;
			sum1 += r[i]*r[i];   // take r*r
			for(int j=0; j<N; j++){
				sum3 += A[i][j]*p[j]; // make A * p
			}
			Ap_vec[i] = sum3;
			sum2 += p[i]*Ap_vec[i]; // take p * (A*p)
		}
		alpha = sum1/sum2; //calculate alpha


		sum1=0;sum2=0;
		//makes needed computations for beta
		for(int i=0; i<N; i++){
			x[i] = x[i] + alpha * p[i];	// find new x
			sum1 += r[i]*r[i];              // calculate r_i * r_i
			r[i] = r[i] - alpha * Ap_vec[i];// find new r
			sum2 += r[i]*r[i];		// calculate r_i+1 * r_i+1
		}
		beta = sum2/sum1;	// calculate beta
		err = sum2;		// error defined as r_i * r_i (note not true residue)
	
		for(int i=0; i<N; i++){
			p[i] = r[i] + beta*p[i]; // find new p
		}

		// true residue
		if(N<100){ // for smaller things, calculate true residue... aribtrarily set at 100
			sum1=0;
			for(int i=0; i<N; i++){
				sum2=0;
				for(int j=0; j<N; j++){
					sum2 += A[i][j]*x[j];
				}
				sum1 += (b[i] - sum2)*(b[i] - sum2);
			}
			err = sum1;
		}
	iter++;
	
	}
	Delete1D(Ap_vec); Delete1D(p); Delete1D(r); //clean up
}


int main(){
	float ** A;
	double * x, * b, tolerance = 1e-6, error=100;
	int N;
	N = 3;
		


	A = Create2D<float >( N, N);
	x = Create1D<double> ( N );
	b = Create1D<double>( N );

	if(N==3){
	// simple test case
		A[0][0] =  2.0 ; A[0][1] = -1.0; A[0][2] =  0.0;
		A[1][0] = -1.0 ; A[1][1] =  2.0; A[1][2] = -1.0;
		A[2][0] =  0.0 ; A[2][1] = -1.0; A[2][2] =  2.0;

		b[0] = 1.0 ; b[1] = 4.0 ; b[2] = 6.0;
	}

	CG(A,x,b,N,tolerance,error);

	cout<<"SERIAL CONJUGATE GRADIENT SOLVER"<<endl;
	cout<<"=== Error ==="<<endl;
	cout<<error<<endl;
	cout<<"===== A ====="<<endl;
	printArray(A[0],N);
	printArray(A[1],N);
	printArray(A[2],N);
	cout<<"===== x ====="<<endl;
	printArray(x,N);
	cout<<"===== b ====="<<endl;
	printArray(b,N);


	//Delete2D(A); Delete1D(x); Delete1D(b);
return 0;
}




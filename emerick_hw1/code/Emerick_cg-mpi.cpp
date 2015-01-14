//// COMPILE /home/sfw/openmpi-1.7.2/bin/mpic++ -std=c++11
//// RUN   /home/sfw/openmpi-1.7.2/bin/mpirun -np ## --hostifle hostfile _____
#include <iostream>
#include <sys/unistd.h>
#include "mpi.h"
#include <vector>
#include <iomanip>
using namespace std;

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



//MPI Testing. Running in parallel many processes:
int main( int argc, char * argv[] ) {


	//initialize MPI
	int rc = MPI_Init( &argc, &argv);
	if ( rc!= MPI_SUCCESS) {
		cout<< "Error starting MPI program. Terminating"<<endl;
		MPI_Abort(MPI_COMM_WORLD,rc);
	}

	double **A, *b, *r, *p, *Ap_vec, *x, *temp;
	double startTime, endTime;
	int N, rank, numnodes, stripSize, numElements, offset, iteration=0, max_iter;
	double sum1=0, sum2=0, stripSizeDec;
	double err = 100, tol;
	double alpha=0, beta=0;
	MPI_Request request1, request2;

	N 	 = 3  ;		// set array dimensions
	max_iter = 100 ;
	tol 	 = 1e-6;


	MPI_Comm_rank( MPI_COMM_WORLD, &rank );     // get current rank
	MPI_Comm_size( MPI_COMM_WORLD, &numnodes ); // find the number of nodes

	stripSizeDec = double(N) / double(numnodes); // size of strip of rows each proc gets
	stripSize = int(stripSizeDec);

	if(stripSize != stripSizeDec || numnodes>N){
		if(rank==0){
			cout<<"N = "<<N<<" for "<<numnodes<<" nodes."<<endl;
		        cout<<"CODE CANNOT YET HANDLE AN ARBITRARY AMOUNT OF PROCESSORS"<<endl;
		}
	
		if(numnodes>N){
			numnodes = N;			
			if(rank==0){
				cout<<"NUMBER OF NODES GREATER THAN THE DIMENSIONS OF ARRAY"<<endl;

			}
		}else{
			if(rank==0){
			cout<<"MUST HAVE N EVENLY DIVISIBLE BY NUMNODES."<<endl;
			}
		}
	MPI_Abort(MPI_COMM_WORLD,rc);
	return 0;
	} 
	MPI_Barrier(MPI_COMM_WORLD);   // make sure != 0 rank nodes don't zoom ahead before check


/////////////////////////////////////////////////////////////////////////////
// CG SOLVING 

	if (rank ==0){ // reserve memory for the arrays... full for 0, 
		A = Create2D<double >(N,N);
		x = Create1D<double >(N);
		b = Create1D<double> (N);
		r = Create1D<double> (N);

		Ap_vec = Create1D<double> (N);
	}  else { // reserve memory for array for the rest of the processors
		A = Create2D<double > (N,stripSize);
		Ap_vec = Create1D<double > (stripSize);   		
  	}


	p = Create1D<double > (N); // p is same size for all processors

	if(rank ==0){ // simple test case
		A[0][0] =  2.0 ; A[0][1] = -1.0; A[0][2] =  0.0;
		A[1][0] = -1.0 ; A[1][1] =  2.0; A[1][2] = -1.0;
		A[2][0] =  0.0 ; A[2][1] = -1.0; A[2][2] =  2.0;

		b[0] = 1.0 ; b[1] = 4.0 ; b[2] = 6.0;
	
		for(int i=0; i<N; i++){// initial conditions
			x[i] = 0.0;
			r[i] = b[i];
			p[i] = b[i];
		}

	}

	if (rank == 0){
		 cout<<"Starting Parallel"<<endl;
		 startTime = MPI_Wtime();
	}

	if (rank ==0){ //distribute A amongst processors by stripSize rows per processors
		numElements = stripSize * N;
		offset = stripSize;
		for(int i=1; i<numnodes; i++){ //comunicate to each of the nodes
			MPI_Isend(A[offset], numElements, MPI_DOUBLE, i, 11, MPI_COMM_WORLD,&request1);
			offset += stripSize;
		}
	} else{
		MPI_Irecv(A[0], stripSize * N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, &request2);
	}
	MPI_Barrier(MPI_COMM_WORLD);
		
	while((err>tol || err<-tol) && iteration<max_iter){ 
		//all processors need p in its entirety
		MPI_Bcast(p,N,MPI_DOUBLE,0,MPI_COMM_WORLD); //broadcast p for the multiplication
		//do the parallel A*p				
		//do parrallel multiplication

		for(int i=0;i<stripSize;i++){ // calculate A * p
			sum1=0.0;
			for(int j=0; j<N; j++){
				sum1 += A[i][j]*p[j];
			}
			Ap_vec[i] = sum1;
		}


		MPI_Barrier(MPI_COMM_WORLD); // wait here to make sure everything caught up
		if (rank!=0){ // region A * p to the rank = 0 processor
			MPI_Isend(&Ap_vec[0],stripSize, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD,&request1);
			
		} else{
			offset = stripSize;
			for(int i=1; i<numnodes; i++){
				MPI_Irecv(&Ap_vec[offset], stripSize, MPI_DOUBLE, i, 13, MPI_COMM_WORLD,&request2);
				offset += stripSize;
			}
		}

	MPI_Barrier(MPI_COMM_WORLD); // wait here to make sure everything received
	   if(rank == 0){

		sum1=0;	sum2=0;
		for(int i =0; i<N; i++){ // calculate values for alpha
			sum1 += r[i]*r[i];		// r*r
			sum2 += p[i]*Ap_vec[i];	        // p*A*p
		}

		alpha = sum1 / sum2 ;

		sum1=0; sum2=0;
		for(int i=0; i<N; i++){
			x[i] = x[i] + alpha*p[i];         // new x
			sum1 += r[i]*r[i];  //r_old squared
			r[i] = r[i] - alpha*Ap_vec[i];	  // new r	
			sum2 += r[i]*r[i];  //r squared
		}	
		beta = sum2/sum1;
		err = sum2; // error is defined as r*r

		for(int i=0; i<N; i++){
			p[i] = r[i] + beta*p[i]; // new p

		}


	   } //end rank=0
	  
	 MPI_Bcast(&err,1,MPI_DOUBLE,0,MPI_COMM_WORLD); // broadacst error to all processors
	
	  // if(rank ==0){
	//	cout<<rank<<": x_"<<iteration<<"=";printArray(x,N);
	  // }

	   iteration++;
	}///// end loop

	if(rank ==0){
		endTime = MPI_Wtime();
		cout<<"PARALLEL CONJUGATE GRADIENT SOLVER"<<endl;
		if(iteration>=max_iter){cout<<"MAX iteration REACHED"<<endl;}
		cout<<"Solution Found:"<<endl;
		cout<<"Took : "<<endTime-startTime<<" s"<<endl;
		cout<<"For : "<<N<<"x"<<N<<" matrix"<<endl;
		cout<<"NumNodes = "<<numnodes<<endl;
		cout<<"In "<<iteration<<" iteration"<<endl;
		cout<<"err "<<err<<endl;

		cout<<"===== A ====="<<endl;
		printArray(A[0],N);
		printArray(A[1],N);
		printArray(A[2],N);
		cout<<"===== x ====="<<endl;
		printArray(x,N);
		cout<<"===== b ====="<<endl;
		printArray(b,N);
	}

	MPI_Finalize();
return 0;
}


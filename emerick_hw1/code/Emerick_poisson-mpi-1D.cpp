//===========================================
// Author: Andrew Emerick
// Homework 1
// 10/11/13
// Parallel Poisson Equation Solver - 1D grid
//===========================================
#include <iostream>
#include <sys/unistd.h>
#include <cmath>
#include <iomanip>
#include "mpi.h"
using namespace std;

// Makes a 1D array with contigious memory
template <class T> T *Create1D(int N1)
{
    T * array = new T  [N1];
    array = new T [N1];
    return array;
};

// Makes a 2D array with contigious memory
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
void printArray(T* a, int N){ //prints an array
	cout<<"#[";
	for(int i=0;i<N;i++){
	
	if(i==0){cout<<setw(2)<<a[i];}
	else{cout<<", "<<setw(2)<<a[i];}	
	}
	cout<<"]"<<endl;
}

// Constructs the Laplacian matrix given a dimension N and the
// dimensions of the grid (i.e. # of grid points in each dimension) n.
// There are currently two blank chuncks placed for the "buffer".. this
// can and should be removed to improve efficiency and memory usage
template <class T>
void fillA(T** A, int Nrow, int Ncol, int n){
	int adjCount=0;
	int triggerMin = 0;
	int triggerMax, cAdjCount=0;
	int shift = n*n;


	for(int i=0; i<Nrow; i++){
		for(int j=0; j<Ncol; j++){
			A[i][j]=0;
		}
	} //preset to all zeroes


	for(int i=0; i<Nrow; i++){
		A[i][i + shift] = 3.0*2.0; // diagonals are 6 (3= number of dimensions)


		// take care of adjacent spots within the same x-dimension
		if(adjCount==0){A[i][i+1 + shift] = -1.0;		adjCount++;}
		else if(adjCount > 0 && adjCount < (n-1)){
			A[i][i+1 + shift] = -1.0; A[i][i-1 + shift]=-1.0;
		adjCount++;
		}
		else{//adjCount = n-1
			A[i][i-1 +shift] = -1.0;
			adjCount = 0;
		}

		// take care of adjacent spots within y
		if(cAdjCount >= n*n){cAdjCount=0;}
		if(cAdjCount<n){A[i][i+n +shift]=-1.0; cAdjCount++;}
		else if(cAdjCount >= n && cAdjCount <(n*n)-n){
			 A[i][i+n+shift] = -1.0;
			 A[i][i-n+shift] = -1.0;
			 cAdjCount ++;
		} else if (cAdjCount >=(n*n) - n && cAdjCount < n*n){
			A[i][i-n+shift] = -1.0;
			cAdjCount++;
		} 
	
		if((i+n*n+shift)<Ncol){A[i][i+n*n+shift]=-1.0;} // take care of adjacent sheets in z
		if((i-n*n+shift)>=0){A[i][i-n*n+shift]=-1.0;}	
	}
}

double densityFunction(double x, double y, double z){
	return -4.0*exp(-2.0*sqrt(x*x + y*y + z*z));
}

// chaged fill A function
// changed A to integers
// resized everything 
// changed fill b

int main( int argc , char * argv[]){
	//initialize MPI
	int rc = MPI_Init( &argc, &argv);
	if ( rc != MPI_SUCCESS) {
		cout <<"# Error starting MPI program. Terminating"<<endl;
		MPI_Abort(MPI_COMM_WORLD,rc);
	}

	int **A; // A can be set with integers even... and should be ..
	double *b, *U, *r, *p, *Ap_vec, *UprintBuffer, *bprintBuffer;
	double h, x, y, z, L, tolerance = 1e-9, error = 1000, maxError = 1000;

	int n, N, element;
	int iterations=0, maxIterations;
	int rank, numnodes, numSheetPerNode, sheetShift;

	double sum1, sum2, sum3, buffer1, buffer2;
	double alpha, beta;

	double startTime, endTime, duration; // timing parameters

	MPI_Request request1, request2;

	MPI_Comm_rank( MPI_COMM_WORLD, &rank);	   // get current rank
	MPI_Comm_size( MPI_COMM_WORLD, &numnodes); // get number of nodes

	MPI_Barrier(MPI_COMM_WORLD);
	
	/// set parameters that should be set later at runtime
		n = 32;
		N = n*n*n;
		maxIterations = n*n*10;
		L = 4.0;				// size of box ... r_bohr = 1
		h = L / (1.0*(n-1));	// step size between grid points
	/// ---------------------------------------------------

	int bufferSize, sizePerProc, totalBuffer, totalSize, factor;

	numSheetPerNode = n/numnodes;		 // needs to be an integer
	bufferSize      = n*n;			 // size of each buffer for sharing b/t faces
	sizePerProc     = n*n*numSheetPerNode; // size of total vector in each node
	totalBuffer     = numnodes*(2*bufferSize);		    // total size of buffer..	.
	totalSize       = N + totalBuffer;				    // total grid points counting buffer

	if ( rank == 0){ // reserve memory for outputs on rank=0 only
		UprintBuffer = Create1D<double > (sizePerProc);
		bprintBuffer = Create1D<double > (sizePerProc);
	}

	// size and construct the arrays
	A = Create2D<int    > (sizePerProc, sizePerProc+ 2*bufferSize);
	U = Create1D<double > (sizePerProc);
	b = Create1D<double > (sizePerProc); 
	r = Create1D<double > (sizePerProc);
	p = Create1D<double > (sizePerProc + 2*bufferSize);
	Ap_vec = Create1D<double > (sizePerProc);

	if (rank == 0){ // output that program is starting
		 cout<<"# Starting Parallel"<<endl;
		 startTime = MPI_Wtime();
	}

	fillA(A,sizePerProc,sizePerProc+2*bufferSize,n); // fill Laplacian matrix for each processor

	// Fill b:
	sheetShift = rank * numSheetPerNode; // shift in z coordinates per processor
	element = 0;
	for(int i=0; i<bufferSize; i++){
		p[i] = 0; p[sizePerProc + i] = 0;
	}

	// fill b 
	for(int k=0; k < numSheetPerNode; k++){
		for(int i=0; i<n; i++){
			for(int j=0; j<n; j++){

				x = (j  - (n-1)/2.0)*h;
				y = (i  - (n-1)/2.0) *h;
				z = ((k+sheetShift) - (n-1)/2.0)*h;
				
				b[element] = densityFunction(x,y,z); //
							
				p[element + bufferSize] = b[element];
				r[element] = b[element];
				U[element] = 0; // initial conditions
				element++;
			}
		}
	
	}// k




	iterations=0;
	// Solve the equation using the conjugate gradient algorthim: 
	while (abs(error)>tolerance && iterations<maxIterations){

//--------------------------- Share between faces ---------------------------------
		//Begin by trading information between faces for the p vector.
		//By design, solving Poisson equation using the conjugate gradient
		//algorithm requires that ONLY p needs to be shared between adjacent faces

		MPI_Barrier(MPI_COMM_WORLD);
		// Send from rank to rank + 1 if rank is even
		if(rank+1 < numnodes && numnodes>1 && rank%2==0){
			MPI_Isend(&p[sizePerProc], n*n, MPI_DOUBLE,
				  rank + 1, 14, MPI_COMM_WORLD, &request1);
			
		} else if (numnodes>1 && rank>0 && rank%2==1){
			MPI_Irecv(&p[0], n*n, MPI_DOUBLE, rank-1, 14, MPI_COMM_WORLD,&request2);

		}
	
		MPI_Barrier(MPI_COMM_WORLD);

		// Send from rank to rank + 1 if rank is odd
		if(rank+1 < numnodes && numnodes>1 && rank%2==1){
			MPI_Isend(&p[sizePerProc], n*n, MPI_DOUBLE,
				  rank + 1, 14, MPI_COMM_WORLD, &request1);
		} else if (numnodes>1 && rank>0 &&rank%2==0){
			MPI_Irecv(&p[0], n*n, MPI_DOUBLE, rank-1, 14, MPI_COMM_WORLD,&request2);
		}
	
	MPI_Barrier(MPI_COMM_WORLD);

	
		// Send from rank to rank - 1 if rank is even
		if(rank-1 >=0 && numnodes>1 && rank%2==0){
			MPI_Isend(&p[bufferSize], n*n, MPI_DOUBLE,
				  rank - 1, 13, MPI_COMM_WORLD, &request1);
			
		} else if (numnodes>1 && rank<numnodes-1 &&rank%2==1){
			MPI_Irecv(&p[sizePerProc+bufferSize], n*n, MPI_DOUBLE,
				  rank+1, 13, MPI_COMM_WORLD, &request2);
			
		}
	
	MPI_Barrier(MPI_COMM_WORLD);
		// Send from rank to rank - 1 if rank is odd
		if(rank-1 >=0 && numnodes>1 && rank%2==1){
			MPI_Isend(&p[bufferSize], n*n, MPI_DOUBLE,
				  rank - 1, 13, MPI_COMM_WORLD, &request1);
			
		} else if (numnodes>1 && rank<numnodes-1 &&rank%2==0){
			MPI_Irecv(&p[sizePerProc + bufferSize], n*n, MPI_DOUBLE,
				  rank+1, 13, MPI_COMM_WORLD, &request2);
			
		}

		MPI_Barrier(MPI_COMM_WORLD);
		// setting boundary conditions (edges must be zeroes)
		if(rank == 0 || rank == numnodes -1){ 

			for(int i=0; i<bufferSize;i++){
				if(rank==0){p[i]=0;}
				else{p[sizePerProc + bufferSize + i] =0;}
			}
		 }

//------------------------ End share between faces ---------------------------- //
// Running conjugate gradient algorithm now for each processors

		for(int i=0; i<sizePerProc; i++){ Ap_vec[i] = 0;} // preset to 0's

		// Loop calculates parameters needed to compute alpha
		sum1=0; sum2=0;
		for(int i = 0; i<sizePerProc; i++){
			sum1 += r[i]*r[i];
			for(int j=0; j<sizePerProc + 2*bufferSize; j++){
				Ap_vec[i] += A[i][j]*p[j];
			}
			sum2+= p[i + bufferSize]*Ap_vec[i];
		}

		// Take the global sums for the numerator and denominator of alpha
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&sum1, &buffer1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&sum2, &buffer2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
		alpha = buffer1 / buffer2; //

		// Compute parameters needed for beta and update U and r
		sum2=0;
		for(int i=0; i<sizePerProc; i++){
			U[i] = U[i] + alpha * p[i + bufferSize];
			r[i] = r[i] - alpha * Ap_vec[i];
			sum2 += r[i]*r[i];
		}
		// global sum for numerator of beta...
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&sum2, &buffer2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
		beta = buffer2 / buffer1;

		error = buffer2; // not the true residue

		// update p on each processor
		for(int i=0; i<sizePerProc; i++){
			p[i + bufferSize] = r[i] + beta*p[i + bufferSize];
		}

		iterations++;
	}//end while loop



//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//	MAKE OUTPUTS
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Status status1,status2; 
	if(rank==0){endTime = MPI_Wtime();
		duration = endTime - startTime;

		
		cout<<setw(14)<<"# Took = "<<setw(10)<<duration<<setw(2)<<" s ("<<duration/60.0<<" min)"<<endl;
		cout<<setw(14)<<"# Error = "<<setw(12)<<error<<endl;
		cout<<setw(14)<<"# n = "<<setw(12)<<n<<endl;
		cout<<setw(14)<<"# N = n*n*n = "<<setw(12)<<N<<endl;

	
	}

	if ( rank ==0 ){ //output U
		cout.setf(ios::scientific);
		cout.precision(5);	
		cout<<"# "<<setw(14)<< "x"
		    <<setw(16)<< "y"
		    <<setw(16)<< "z"
		    <<setw(16)<< "U(x,y,z)"
		    <<setw(16)<< "b(x,y,z)" << endl;


		for(int i=0; i<numnodes;i++){


		   if(i>0){

			MPI_Recv(&UprintBuffer[0], sizePerProc, MPI_DOUBLE,
				 i, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Wait(&request1,&status1);

			MPI_Recv(&bprintBuffer[0], sizePerProc, MPI_DOUBLE,
				 i, 19, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Wait(&request1,&status1);

			
		   }else{ 
			for(int xi=0; xi<sizePerProc; xi++){
				UprintBuffer[xi] = U[xi]; 	
		   		bprintBuffer[xi] = b[xi];
			}
	           }

			element = 0; sheetShift = i*numSheetPerNode;
			for(int k=0; k < numSheetPerNode; k++){
				for(int yi=0; yi<n; yi++){
					for(int j=0; j<n; j++){

					x = (j  - (n-1)/2.0)*h;
					y = (yi  - (n-1)/2.0) *h;
					z = ((k+sheetShift) - (n-1)/2.0)*h;
										
					cout<<setw(16)<< x 
					    <<setw(16)<< y
					    <<setw(16)<< z
					    <<setw(16)<< UprintBuffer[element]
					    <<setw(16)<< bprintBuffer[element] << endl;
					

					element++;
					}
				}
	
			}// k


		
			}
		



	}else{	

		MPI_Send(&U[0], sizePerProc, MPI_DOUBLE,
			 0, 17, MPI_COMM_WORLD);

		MPI_Wait(&request2,&status2);

		MPI_Send(&b[0], sizePerProc, MPI_DOUBLE,
			 0, 19, MPI_COMM_WORLD);
		MPI_Wait(&request2,&status2);


	}


	MPI_Barrier(MPI_COMM_WORLD);

	
	MPI_Finalize();
	return 0;
}

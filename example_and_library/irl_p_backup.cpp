//===========================================
// Author: Andrew Emerick
// Homework 2
// 11/22/13
// 
// Serial irl solver
//// COMPILE /home/sfw/openmpi-1.7.2/bin/mpic++ -std=c++11
// mpic++ -std=c++11 -o irl_p.exe irl_p.cpp -I /home/sfw/include -L /home/sfw/lib -lgsl -lgslcblas
//// RUN   /home/sfw/openmpi-1.7.2/bin/mpirun -np ## --hostfile hostfile _____
// mpirun -np 1 --hostfile hostfile irl_p.exe
//===========================================
#include<iostream>
#include <sys/unistd.h>
#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include "mpi.h"
#include "irl_p.h"
//#include <vector>
//-------------------------------------
// Include things needed for GSL

// Needed for the eigensolver
#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
// Needed for the QR decomposition
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
//-------------------------------------
using namespace std;

/* Use explicit orthogonalization */
#define LANCZOS_ORTHO 1
#define GLOBAL_L 40

//int main(){
    // CODE FOR openmp-ness
int main( int argc , char * argv[]){

    int rc = MPI_Init( &argc, &argv);     
    if ( rc!= MPI_SUCCESS) {
        cout <<"Error starting MPI progam. Terminating."<<endl;
        MPI_Abort(MPI_COMM_WORLD,rc);
    }
    
    int rank, numnodes;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &numnodes);

    
    int    global_Lx, global_Ly, global_Lz;
    double global_V;
    vector<double > rloc;
    vector< vector<int > > p_xyz;
    global_Lx = GLOBAL_L; global_Ly = global_Lx; global_Lz = global_Lx;    
    global_V  = global_Lx * global_Ly * global_Lz;
    
    // Set the center coordinates
    zeros1D(rloc,3);
    rloc[0] = global_Lx/2.0 - 3.2;
    rloc[1] = global_Ly/2.0 - 1.6; 
    rloc[2] = global_Lz/2.0 + 2.7;

    // fills matrix associating the lexical rank to a px,py,pz grid of
    // processors. Used to convert local to global coordinates for 
    // the set_nbr_bc function. Set to be even 3D grid
    zeros2D(p_xyz,numnodes,9); 
    set_p_xyz(p_xyz,numnodes,numnodes,numnodes);


    // Now worrying about local things mainly
    MPI_Barrier( MPI_COMM_WORLD);
    int buffer_size;
    int Lx, Ly, Lz, V;
    int edet, kmax, pmax, kpmax;
    int nrestart;
    vector< vector<float > > nbr;
    vector<int> bc;
    double tol;
    vector<double > gridVals;    


    double a, tmp_scalar=0;
    
    vector< vector<double > > v;
    int eknown;
    vector<double > col_vec1,col_vec2;
    vector<double > evalue, alpha, beta;
    vector< vector<double > > evector;
    vector< vector<double > > H, dp;
    vector< vector<double > > VH, DH;
    vector< vector<double> > tmp_vector;
    vector< vector<double> > tmp_vector2;


    // needed for QR decomp and shifts
    vector< vector<double> > Q, Qt, R, Hplus, vp;
    double betahat, betap, betatilde;    
    vector<double> fpk;

    double norm;
    double sum=0;
    
    zeros1D(gridVals,4);

    
    // Set Lanczos-y parameters
    edet = 2 ; // Number of eigenvalues to find
    kmax = 6; // Number of eigenvalues / eigenvectors
    pmax = 15; // Number of QS decompositions (a.k.a. shifts)
    kpmax = kmax + pmax;
    
       // Desired accuracy for resolved eigenvalues
    tol = 1.0E-8; 
    nrestart = 100;
    
       
    // Set grid related parameters
    Lx = global_Lx / (1.0*numnodes);
    Ly = global_Ly / (1.0*numnodes); 
    Lz = global_Lz / (1.0*numnodes);  
    V = Lx*Ly*Lz;
    buffer_size = 6 * Lx*Lx; // NEED TO CHANGE IF LX!=LY!=LZ

    gridVals[0] = Lx; gridVals[1] = Ly; gridVals[2] = Lz;
    
    // Location of center of the atom

    
    // set lattice spacing in appropriate units
    a = 0.5; // should be 0.5????
    gridVals[3] = a;
    
    // Declare the nearest neighbors function:
    // 0=xp 1=xm 2=yp 3=ym 4=zp 5=zm 6=local 1/r term
    zeros2D(nbr,V,7);
        
    // Make the Lanczos vectors
    zeros2D(v ,V + buffer_size ,kpmax+1);
    
    // Resolved eigenvals and vecs to save
    eknown  = 0;
    zeros1D(evalue , kmax);
    zeros2D(evector, V,kmax);
    
    // Lanczos paramaters and matrix
    zeros1D(alpha, kpmax);
    zeros1D(beta , kpmax);
    zeros2D(H    , kpmax,kpmax );
    zeros2D(VH,kpmax,kpmax);
    zeros2D(DH,kpmax,kpmax);
    zeros2D(dp,kpmax,kpmax);
    
    zeros1D(fpk,V);

    zeros2D(vp,V,kpmax+1);

    zeros2D(Hplus,kpmax,kpmax);
    zeros2D(Q,kpmax,kpmax);
    zeros2D(Qt,kpmax,kpmax);
    zeros2D(R,kpmax,kpmax);

    cout<<"Setting bc"<<endl;
    // set boundary conditions
    set_bc_nbr(nbr,bc,gridVals,rloc,p_xyz,rank,V);
    cout<<"Set bc"<<endl;

    // Set random starting vector
    for (int i=0;i<V;i++){
        v[i][0] = rand() - 0.5;
    }

    // enforce boundary conditions
    for (int i=0;i<bc.size();i++){
	    v[bc[i]][0] = 0.0; 
    }    
    sum = 0;
    for(int i=0; i<V; i++){
        sum += v[i][0]*v[i][0];
    }
    norm = sqrt(sum);
    for (int i=0; i<V;i++){
	    v[i][0] = v[i][0]/norm;	
    }
	
    cout<<"Lanczos"<<endl;
    // Do the initial Lanczos step
    lanczos(0,kpmax-1,v,alpha,beta,nbr,bc, p_xyz, V, buffer_size, rank, numnodes);
    cout<<"End Lanczos"<<endl;
    //for(int i=1699; i<1721; i++){
    //    cout<<v[i][0]<<endl;
    // }


    // Loop over the number of restart steps
    for(int nr=0; nr<nrestart; nr++){



        // Find eigenvalues and eigenvectors of H
        for(int k=0; k<kpmax; k++){
            H[k][k] = alpha[k];
            if( k > 0){
                H[k-1][k  ] = beta[k-1];
                H[k  ][k-1] = beta[k-1];
            }	    
        }

        // use the eigenvalue solver to get VH and DH
        // need to construct VH and DH as kpmax x kpmax matrices
        // VH = eigenvectors of H
        // DH = eigenvalues of H in a diagonal matrix
        eigensolver(H,kpmax,VH,DH);


        // Check the convergence of eigenvalues
        // if converged, save eigenvector and eigenvalues

        for( int k=0; k < kpmax; k++){

            tmp_scalar = abs( beta[kpmax-1] * VH[kpmax-1][k] / DH[k][k]);  

            if(  k < kmax){
                printf("Eigenvalue number %d, value %e, fractional accuracy %e\n",k,
                                              DH[k][k],tmp_scalar);
            }     
            if( (tmp_scalar < tol) && (k >  eknown - 1 ) && (k < kmax ) ){
                eknown = eknown + 1;
                evalue[eknown-1] = DH[k][k];
                for( int mm = 0; mm < V; mm++){
                    for (int kk =0; kk < kpmax; kk++){
                        evector[mm][eknown-1] = evector[mm][eknown-1]
                                                + v[mm][kk] * VH[kk][k];
                                
                    }	                    
                }
        

                sum = 0;
                for(int iii=0; iii <evector.size(); iii++){
                    sum += evector[iii][eknown-1]*evector[iii][eknown-1];
                }

                norm = sqrt(sum);

                for(int j=0; j < evector.size(); j++){
                   evector[j][eknown-1] = evector[j][eknown-1] / norm;
        
                }
             } //end if
    
        } // end k->kpmax convergence loop


        if( eknown < edet){

        // Do pmax shifts
 
            zeros2D(Q,kpmax,kpmax);
            for(int i=0; i<kpmax; i++){
                Q[i][i] = 1.0;
            }
            matrix_copy(Hplus,H);

    //  cout<<"matrix copy success to tmp_vector"<<endl;
            for(int k = kmax ; k < kpmax; k++){

                zeros2D(tmp_vector,kpmax,kpmax);
                matrix_copy(tmp_vector,Hplus);
                zeros2D(Qt,kpmax,kpmax);
                zeros2D(R,kpmax,kpmax);
                // make some temporary matrix
                for(int kk=0; kk<kpmax; kk++){
                     tmp_vector[kk][kk] = tmp_vector[kk][kk] - DH[k][k]; // along diagonal
                }

                qrdecomp(tmp_vector,Qt,R,kpmax); // does a QR decomposition of A

                sum=0;
                // multiply Q by Qt
                for(int i=0; i<kpmax; i++){
                    for(int j=0; j<kpmax; j++){
                        for(int jj=0; jj<kpmax; jj++){
                            sum += Q[i][jj] * Qt[jj][j];
		                }
                    
                        tmp_vector[i][j] = sum;
                        sum=0;
                    }
                }//done Q*Qt  

                // copy tmp over to qt
                matrix_copy(Q,tmp_vector);

                // multiply Qt^T * Hplus * Qt
                sum=0;
                // multiply Q by Qt
                for(int i=0; i<kpmax; i++){
                    for(int j=0; j<kpmax; j++){
                        for(int jj=0; jj<kpmax; jj++){
                            sum += Hplus[i][jj] * Qt[jj][j];
            		    }
                    
                    tmp_vector[i][j] = sum;
                    sum=0;
                    }
                }

                zeros2D(tmp_vector2,kpmax,kpmax);

                sum=0;
                // multiply Qt^T * (HQt)
                for(int i=0; i<kpmax; i++){
                    for(int j=0; j<kpmax; j++){
                        for(int jj=0; jj<kpmax; jj++){
                            sum += Qt[jj][i] * tmp_vector[jj][j];
            		    }
                    
                    tmp_vector2[i][j] = sum;
                    sum=0;
                    }
                }                    

	        matrix_copy(Hplus,tmp_vector2);
            
                   
        } // k loop kmax -> kpmax for QR decomp
    

        // multiply v and q to get vp
        sum=0;
        for (int i = 0 ; i < V ; i++){
            for(int j = 0 ; j < kpmax ; j++){
                for(int jj = 0 ; jj < kpmax  ; jj++){
                   sum += v[i][jj] * Q[jj][j];
                }                 
                vp[i][j] = sum;
                sum=0;
            }
        }   

        betahat = Hplus[kmax-1][kmax];
        betatilde = beta[kpmax-1] * Q[kpmax-1][kmax-1];

        // calculate fpk
        for(int i=0; i<V; i++){
            fpk[i] = betahat*v[i][kmax] + betatilde * v[i][kpmax];
    	}

        sum =0;
        for(int iii=0; iii< fpk.size(); iii++){
            sum += fpk[iii]*fpk[iii];
        }
        betap = sqrt(sum);

        // normalize fpk
        for (int i=0; i<fpk.size(); i++){
            fpk[i] = fpk[i]/betap;
        }

        // Check orthoginality

        for(int k1=1;k1<kpmax; k1++){
            for (int k2=0; k2<=k1-1; k2++){ //MAY NEED TO CHANGE FROM <= TO <
      
                sum = 0;
                for(int iii=0; iii < V; iii++){
                    sum += vp[iii][k1]*vp[iii][k2];
                }
                dp[k1][k2] = sum;

             }

             sum=0;
             for(int iii=0; iii<V; iii++){
                 sum += vp[iii][k1] * fpk[iii];
             }
             dp[k1][kpmax-1] = sum;
        }

        printf("Maximum dot product for v+ vectors is %e\n",max_val_matrix(dp));
        
        // Now restart Lanczos .....

        printf("Implicit restart %d of Lanczos. %d eigenvalues accurately determined\n",
               nr,eknown);

        eknown = 0.0;
        
        // copy vp into v
        for(int i=0; i<V; i++){
            for( int j=0; j<kmax; j++){
                v[i][j] = vp[i][j];
            } 
        }

        // set alpha and beta to zeros
        zeros1D(alpha,kpmax);
        zeros1D(beta,kpmax);

        // assign new alpha and beta
        for(int k=0; k<kmax; k++){
            alpha[k] = Hplus[k][k];
            if (k > 0){
                beta[k-1] = Hplus[k-1][k];
            }
        }

        lanczos(kmax-1,kpmax-1,v,alpha,beta,nbr,bc,p_xyz,V,buffer_size,rank,numnodes);

    }else{ // end eknown < edet if
        break; 
    }

} // end number of restart steps loop


// check eigenvectors
for(int k=0; k<edet; k++){

    for(int i=0; i<V;i++){
        v[i][k] = evector[i][k];
    }

    share_v(v, p_xyz, k, V, buffer_size, rank, numnodes);
    Av(k,k+1, v, nbr, bc, V);

    sum=0;
    for(int iii=0; iii<V; iii++){
         sum += v[iii][k] * v[iii][k+1];
    }
    tmp_scalar = sum;

    sum=0;
    for(int iii=0; iii<evector.size(); iii++){
        sum += evector[iii][k]*evector[iii][k];
    }
    norm = sqrt(sum);

    printf("Eigenvalue %d = %e, norm %e, vAv for eigenvector = %e \n", k,
            evalue[k], norm, tmp_scalar);
            
    for(int kk=0; kk<= k-1; kk++){ 
        sum = 0;
        for(int iii=0; iii<evector.size(); iii++){
            sum += evector[iii][k] * evector[iii][kk];
        }
       
        dp[k][kk] = sqrt(sum);
	    
    }

}// end the check eigenvectors loop

tmp_scalar = max_val_matrix(dp);
printf("Maximum dot product for eigenvectors with k~=kk is %e \n", tmp_scalar);
return 0;
}
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// End Main
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// need to pass alpha, beta, v, nbr, bc
void lanczos(int k1, int k2,
             vector< vector<double> > & v, vector<double> & alpha,
             vector<double> & beta, const vector< vector<float> > & nbr,
             const vector<int> & bc, const vector< vector<int> > & p_xyz,
             const int V,
             const int bufferSize, const int rank, const int numnodes){

    double dp;


    // assuming k1 and k2 translate as is from MATLAB indexing...
    // changing all indeces from MATLAB to C++ by subtracting one
    // from the MATLAB versions.....
    for(int i = k1; i <= k2 ; i++){

        share_v(v, p_xyz, i, V, bufferSize, rank, numnodes);
        Av(i,i+1,v,nbr,bc, V);


        dp = 0;

        // &*&* THIS NEEDS TO BE GLOBALLY summed (DP)... used in alpha
        for(int iii=0; iii<V; iii++){
            dp += v[iii][i]*v[iii][i+1];
        }
        
        alpha[i] = dp ; 
        if ( i == 0){ 
            for(int j=0; j < V; j++){
                v[j][i+1] = v[j][i+1] - alpha[i] * v[j][i];
            }    
        }else{
            for(int j=0; j < V; j++){
                v[j][i+1] = v[j][i+1] - alpha[i]*v[j][i] - beta[i-1] * v[j][i-1];
            }
        }
    
        // Explicit orthongonalizatioin
        
        if( LANCZOS_ORTHO ==1){
            for (int ii=0 ; ii <= i; ii++){  
            
                    dp = 0;
                    for(int iii=0; iii<V; iii++){
                        dp += v[iii][ii] * v[iii][i+1];
                    }
                    

                for( int k=0; k<V; k++){
                      v[k][i+1] = v[k][i+1] - dp * v[k][ii];
                }
            
            }        
        } // end ortho check
        
        // &*&* THIS NEEDS TO BE GLOBALLY summed (DP)... used in beta
        dp = 0;
        for( int iii=0; iii<V; iii++){
            dp += v[iii][i+1]*v[iii][i+1];
        }
        beta[i] = sqrt(dp);

        // does not need global communication
        for(int j=0; j< V; j++){
            v[j][i+1] = v[j][i+1] / beta[i];
        }


    }// end k1 -> k2 loop

}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Applies matrix times vector
void Av(int i, int j, vector< vector<double> > & v, const vector< vector<float> > & nbr,
        const vector<int> & bc, const int V){

    for(int n = 0; n < V ; n++){
        v[n][j] = (3.0 - nbr[n][6] ) * v[n][i]; // 1/r term for self
        for(int m=0 ; m < 6; m++){ // m < 7

            v[n][j] = v[n][j] - 0.5 * v[ nbr[n][m] ][i];            
        }   
    }

    // apply the boundary conditions
    for (int n = 0; n < bc.size(); n++){
        v[ bc[n] ][j] = 0.0;    
    }

}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
void share_v(vector< vector<double > > & v, const vector< vector<int > >& p_xyz,
             const int kshare, // the kth vector of v to share
             const int V, const int buffer_size,
             const int rank, const int numnodes){
    if(numnodes == 1){ return; } // no sharing needed for single node
    int pxp,pxm,pyp,pym,pzp,pzm;    
    // 3 = px+   4 = px-
    // 5 = py+   6 = py-
    // 7 = pz+   8 = pz-
    pxp = p_xyz[rank][3]; pxm = p_xyz[rank][4];
    pyp = p_xyz[rank][5]; pym = p_xyz[rank][6];
    pzp = p_xyz[rank][7]; pzm = p_xyz[rank][8];

    // MPI things
    MPI_Request request1, request2;

    // Declare buffers
    vector<double > sendBuffer1, recvBuffer1; // send xp
    vector<double > sendBuffer2, recvBuffer2; // send xm
    vector<double > sendBuffer3, recvBuffer3;
    vector<double > sendBuffer4, recvBuffer4;
    vector<double > sendBuffer5, recvBuffer5;
    vector<double > sendBuffer6, recvBuffer6;

    // Initialize buffers
    zeros1D(sendBuffer1,buffer_size); zeros1D(recvBuffer1,buffer_size);
    zeros1D(sendBuffer2,buffer_size); zeros1D(recvBuffer2,buffer_size);
    zeros1D(sendBuffer3,buffer_size); zeros1D(recvBuffer3,buffer_size);
    zeros1D(sendBuffer4,buffer_size); zeros1D(recvBuffer4,buffer_size);
    zeros1D(sendBuffer5,buffer_size); zeros1D(recvBuffer5,buffer_size);
    zeros1D(sendBuffer6,buffer_size); zeros1D(recvBuffer6,buffer_size);

    // Fill the send buffers from v
    for(int i=0; i<buffer_size; i++){
        sendBuffer1[i] = v[V + i + buffer_size*0.0][kshare];
        sendBuffer2[i] = v[V + i + buffer_size*1.0][kshare];
        sendBuffer3[i] = v[V + i + buffer_size*2.0][kshare];
        sendBuffer4[i] = v[V + i + buffer_size*3.0][kshare];
        sendBuffer5[i] = v[V + i + buffer_size*4.0][kshare];
        sendBuffer6[i] = v[V + i + buffer_size*5.0][kshare];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards positive x
    MPI_Isend(&sendBuffer1[0], buffer_size, MPI_DOUBLE,
              pxp, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer1[0], buffer_size, MPI_DOUBLE,
              pxm, pxm, MPI_COMM_WORLD, &request1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards negative x
    MPI_Isend(&sendBuffer2[0], buffer_size, MPI_DOUBLE,
              pxm, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer1[0], buffer_size, MPI_DOUBLE,
              pxp, pxp, MPI_COMM_WORLD, &request1); 

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards positive y
    MPI_Isend(&sendBuffer3[0], buffer_size, MPI_DOUBLE,
              pyp, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer3[0], buffer_size, MPI_DOUBLE,
              pym, pym, MPI_COMM_WORLD, &request1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards negative y
    MPI_Isend(&sendBuffer4[0], buffer_size, MPI_DOUBLE,
              pym, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer4[0], buffer_size, MPI_DOUBLE,
              pyp, pyp, MPI_COMM_WORLD, &request1); 

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards positive z
    MPI_Isend(&sendBuffer5[0], buffer_size, MPI_DOUBLE,
              pzp, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer5[0], buffer_size, MPI_DOUBLE,
              pzm, pzm, MPI_COMM_WORLD, &request1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards negative zs
    MPI_Isend(&sendBuffer6[0], buffer_size, MPI_DOUBLE,
              pzm, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer6[0], buffer_size, MPI_DOUBLE,
              pzp, pzp, MPI_COMM_WORLD, &request1); 

    MPI_Barrier(MPI_COMM_WORLD);

    // Loop through and assign to v buffers    
    for(int i=0; i<buffer_size; i++){
        v[V + i + buffer_size*0.0][kshare] = recvBuffer1[i];
        v[V + i + buffer_size*1.0][kshare] = recvBuffer2[i];
        v[V + i + buffer_size*2.0][kshare] = recvBuffer3[i];
        v[V + i + buffer_size*3.0][kshare] = recvBuffer4[i];
        v[V + i + buffer_size*4.0][kshare] = recvBuffer5[i];
        v[V + i + buffer_size*5.0][kshare] = recvBuffer6[i];
    }
  
    MPI_Barrier(MPI_COMM_WORLD);
}
// ----------------------------------------------------------------------------
// set_bc_nbr loads the neighbor array with the lexical index for the neighbors
// of site n and determines whther site n is on the boundary or not
//
// Requires calling function to pass a predeclared and properly sized matrix 
// nbr where the neighbors will be assigned. In addition, the grid dimensions 
// are required, along with the location of the center of the atom, and the 
//  conversition to real units
void set_bc_nbr(vector< vector<float > > & nbr, vector<int > & bc,
                const vector<double > & gridDimensions, 
                const vector<double > & center,
                const vector< vector<int> > & p_xyz, const int rank, 
                const int V){

    int Lx,Ly,Lz;
    int  x, y, z;
    int xp, yp, zp, xm, ym, zm;
    int global_x, global_y, global_z; // global x y z
    int n;
    int fs;

    double a = gridDimensions[3]; // the unit conversion for grid sizes

    Lx = gridDimensions[0]; Ly = gridDimensions[1]; Lz = gridDimensions[2];
    
    fs = Lx * Lx ; // assumes Lx=Ly=Lz

    for (int x=0; x<Lx; x++){
        for(int y=0; y<Ly; y++){
            for(int z=0; z<Lz; z++){
                // calculate the global values
                global_x = x + Lx * p_xyz[rank][0];
                global_y = y + Ly * p_xyz[rank][1];
                global_z = z + Lz * p_xyz[rank][2]; 
            
                // x minus
/*
                     if( x == 0 && global_x == 0){global_bndry = true;}
                else if( x == 0 ){ xm = V + (y + Ly*z) + 0.0*fs;}
                else{              xm = x-1;}

                // x plus
                if(x == Lx-1 && global_x == GLOBAL_L-1){global_bndry=true;}
                else if( x == Lx-1 ){ xp = V + (y + Ly*z) + 1.0*fs;}
                else{                 xp = x+1;}

                // y minus
                     if( y == 0 && global_y == 0){global_bndry = true;}
                else if( y == 0 ){ ym = V + (x + Lx*z) + 2.0*fs;}
                else{              ym = y-1;}

                // y plus
                if(y == Ly-1 && global_y == GLOBAL_L-1){global_bndry=true;}
                else if( y == Ly-1 ){ yp = V + (x + Lx*z) + 3.0*fs;}
                else{                 yp = y+1;}

                // z minus
                     if( z == 0 && global_z == 0){global_bndry = true;}
                else if( z == 0 ){ zm = V + (x + Lx*y) + 4.0*fs;}
                else{              zm = z-1;}

                // z plus
                if(z == Lz-1 && global_z == GLOBAL_L-1){global_bndry=true;}
                else if( z == Lz-1 ){ zp = V + (x + Lx*y) + 5.0*fs;}
                else{                 zp = z+1;}
*/        
                // Set the boundaries to read from the buffers
                if( x == 0 )   { xm = V + (y + Ly*z) + 0.0*fs;} else{ xm = x-1;}
                if( x == Lx-1 ){ xp = V + (y + Ly*z) + 1.0*fs;} else{ xp = x+1;}
  
                if( y == 0 )   { ym = V + (x + Lx*z) + 2.0*fs;} else{ ym = y-1;}
                if( y == Ly-1 ){ yp = V + (x + Lx*z) + 3.0*fs;} else{ yp = y+1;}
                
                if( z == 0 )   { zm = V + (x + Lx*y) + 4.0*fs;} else{ zm = z-1;}
                if( z == Lz-1 ){ zp = V + (x + Lx*y) + 5.0*fs;} else{ zp = z+1;}
        
                n = get_n(x,y,z,gridDimensions);
                nbr[n][0] = get_n(xp,  y,  z, gridDimensions);
                nbr[n][1] = get_n(xm,  y,  z, gridDimensions);
                nbr[n][2] = get_n( x, yp,  z, gridDimensions);
                nbr[n][3] = get_n( x, ym,  z, gridDimensions);
                nbr[n][4] = get_n( x,  y, zp, gridDimensions);
                nbr[n][5] = get_n( x,  y, zm, gridDimensions);
                // Set last element to distance from center of atom
                nbr[n][6] = a / 
                           sqrt(  (global_x-center[0])*(global_x-center[0])   +
                                  (global_y-center[1])*(global_y-center[1])   +
                                  (global_z-center[2])*(global_z-center[2])  ) ;
                                
                // if on a global boundary
                if(global_x == 0 || global_y == 0 || global_z == 0 ||
                   global_x == GLOBAL_L - 1 || 
                   global_y == GLOBAL_L - 1 ||
                   global_z == GLOBAL_L - 1    ){
                   
                    bc.push_back(n);
                }
            }   
        }
    } 

}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Returns the local lexical order of index n for a point x,y,z
int get_n(int x, int y, int z, const vector<double > & gridDimensions){
    int Lx,Ly,Lz;
    int V;
    
    Lx = gridDimensions[0]; Ly = gridDimensions[1]; Lz = gridDimensions[2];
    V = Lx*Ly*Lz;

    if( x >= V || y >= V || z >= V ){
        // returns maximum of x y or z
        return  z>( (x<y)?y:x ) ? z : ( (x<y)?y:x )  ;

    }else{
        return x + Lx*(y) + Lx * Ly * (z);
    }
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Solves the matrix H for its eigenvalues and eigenvectors and returns H
// as VH and DH, vectors of vectors containing the eigenvectors and diagonal
// of eigenvalues respectively.
// This is really a wrapper for the eigen.c GSL example to take a matrix 
// defined using vectors, and return matrices defined using vectors...
// this abstracts the GSL stuff from the main program
void eigensolver(const vector< vector<double > > & H, int kpmax,
                 vector< vector<double > > & VH, vector< vector<double > > & DH){
    /*double *data = new double [kpmax*kpmax];
    int counter = 0;    
   
    // Need to flatten the H matrix
    for(int i=0; i<kpmax; i++){
        for(int j=0; j<kpmax; j++){
            data[counter] = H[i][j];
            counter ++;
        }
    }*/

    gsl_matrix *m = gsl_matrix_alloc (kpmax,kpmax);

    for(int i=0; i<kpmax; i++){
        for(int j = 0; j < kpmax; j++){
            gsl_matrix_set(m,i,j,H[i][j]);
        }
    }

   // gsl_matrix_view m = gsl_matrix_view_array (data,kpmax,kpmax); // gsl matrix
    gsl_vector *eval = gsl_vector_alloc (kpmax);
    gsl_matrix *evec = gsl_matrix_alloc (kpmax,kpmax);

    gsl_eigen_symmv_workspace *w =
        gsl_eigen_symmv_alloc (kpmax);

    gsl_eigen_symmv(m,eval,evec,w);
    //gsl_eigen_symmv (&m.matrix, eval, evec, w);
    gsl_eigen_symmv_free (w);
 
    gsl_eigen_symmv_sort(eval,evec,
                         GSL_EIGEN_SORT_VAL_ASC);

    // Extract eigenvalues  and assign along diagonal of DH
    // Extract eigenvectors and assign along columns  of VH
    for(int i=0; i<kpmax; i++){
        DH[i][i] = gsl_vector_get (eval,i);
                 
        //gsl_vector_view evec_i = gsl_matrix_column (evec,i);
        for(int j=0; j<kpmax; j++){
            VH[j][i] = gsl_matrix_get(evec,j,i);
        }
    }

    gsl_vector_free (eval);
    gsl_matrix_free (evec);
    gsl_matrix_free (m);
   // cout<<"end eigensolver"<<endl;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Returns Q and R from a decomposition
// This is a wrapper around the GSL Library's QR decomposition functionality
// that allows passing and returning of vectors... abstracts the GSL-ness
// from the main program
void qrdecomp(const vector< vector<double > > & Avec, vector< vector<double > > & Qvec,
              vector< vector<double > > & Rvec, int kpmax){
   // cout<<"making matrices"<<endl;
    gsl_matrix *A, *Q, *R;
    gsl_vector *A_tau;
  // cout<<"memory assigning Q"<<endl;

    Q = gsl_matrix_alloc(kpmax,kpmax);
   // cout<<"memory assigning R"<<endl;
    R = gsl_matrix_alloc(kpmax,kpmax);
  //  cout<<"memory assigning A"<<endl;
    A = gsl_matrix_alloc(kpmax,kpmax);
  //  cout<<"memory assigning Atau"<<endl;
    A_tau = gsl_vector_alloc(kpmax);


    for(int i=0; i<kpmax; i++){
        for(int j=0; j<kpmax; j++){
            gsl_matrix_set(A,i,j,Avec[i][j]);
        }
    }

   // cout<<"solving"<<endl;
    gsl_linalg_QR_decomp(A, A_tau);
    gsl_linalg_QR_unpack(A, A_tau, Q, R);
 //   cout<<"solved"<<endl;
    for(int i = 0 ; i< kpmax ; i++){
        for(int j = 0 ; j< kpmax ; j++){
            Qvec[i][j] = gsl_matrix_get(Q,i,j);
            Rvec[i][j] = gsl_matrix_get(R,i,j);
        }
    }
   // cout<<"problem freeing"<<endl;
    gsl_matrix_free(A);
    gsl_matrix_free(Q);
    gsl_matrix_free(R);
    gsl_vector_free(A_tau);
  //  cout<<"End qr decomp"<<endl;
}

// ----------------------------------------------------------------------------
// Sets the p_xyz matrix, which associates the lexical rank value assigned to
// each processor to a px,py,pz coordinates, which can be used to convert
// local (x,y,z) to global (x,y,z)
void set_p_xyz(vector< vector<int > > & p_xyz, int Nx, int Ny, int Nz){
    int p;
    int pxp, pyp, pzp;
    int pxm, pym, pzm;

    for(int px = 0; px < Nx; px++){
        for(int py = 0; py < Ny; py++){
            for(int pz = 0; pz < Nz; pz++){

                if ( px == Nx - 1){ pxp = 0;} else{ pxp = px +1;}
                if ( py == Ny - 1){ pyp = 0;} else{ pyp = py +1;}
                if ( pz == Nz - 1){ pzp = 0;} else{ pzp = pz +1;}
                if ( px == 0     ){ pxm = Nx-1;} else{ pxm = px - 1;}
                if ( py == 0     ){ pym = Ny-1;} else{ pym = py - 1;}
                if ( pz == 0     ){ pzm = Nz-1;} else{ pzm = pz - 1;}

                p = get_p(px,py,pz,Nx,Ny,Nz);

                p_xyz[p][0] = px;
                p_xyz[p][1] = py;
                p_xyz[p][2] = pz;
                p_xyz[p][3] = get_p( pxp,  py,  pz,Nx,Ny,Nz);
                p_xyz[p][4] = get_p( pxm,  py,  pz,Nx,Ny,Nz);
                p_xyz[p][5] = get_p(  px, pyp,  pz,Nx,Ny,Nz);
                p_xyz[p][6] = get_p(  px, pym,  pz,Nx,Ny,Nz);
                p_xyz[p][7] = get_p(  px,  py, pzp,Nx,Ny,Nz);
                p_xyz[p][8] = get_p(  px,  py, pzm,Nx,Ny,Nz);
            }
        }
    } 
}
// ----------------------------------------------------------------------------
// Returns lexical order p for the processors (corresponds to rank)
//
int get_p(const int px, const int py, const int pz,
           const int Nx, const int Ny, const int Nz){
    return px + Nx * py + Nx*Ny*pz;
}


// ----------------------------------------------------------------------------
// Resiszes a 2D vector with zeros
template <class T> 
void zeros2D(vector< vector<T > > & v, int N1, int N2)
{
    // N1 = number of columns
    // N2 = number of rows
    if(v.size() == N1 && v[0].size() == N2){
        for(int i=0; i<N1; i++){
            for(int j=0; j<N2; j++){
                v[i][j] = 0.0;
            }
         }
     }else{

        v.resize( N1, vector<T >(N2,0.0) );
     }
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Resizes a vector with zeros
template <class T>
void zeros1D(vector<T > & v, int N1)
{
    v.resize(N1,0.0);
}
// ----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// copies a matrix to another matrix
template <class T>
void matrix_copy(vector< vector<T > > & copied, 
                 const vector< vector<T > > & orig){
    int n = copied.size();
    int m = copied[0].size();

    for(int i=0;i<n;i++){
        for(int j=0; j<m;j++){
            copied[i][j] = orig[i][j];
        }
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Returns the maximum value in a matrix
double max_val_matrix(const vector< vector<double > > & x){
    double maximum = -1e99;

    for(int i=0; i<x.size(); i++){
        for(int j=0; j<x.size(); j++){
            if(x[i][j]>maximum){ maximum = x[i][j];}
        }
    }

    return maximum;
}




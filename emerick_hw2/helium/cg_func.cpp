#include "mpi.h"
#include <vector>
#include <cmath>
using namespace std;
#include "cg_func.h"
#include "parallel_functions.h"
#define GLOB_A 1.0

// -----------------------------------------------------------------------------
// Congugate gradient algorthim to find the solution of Ax=b where A is some
// large sparse matrix... aka laplacian...
// -----------------------------------------------------------------------------
void cg_func(vector<double> & b, vector<double> & final_U,
             const vector< vector<float> > nbr,
             const vector< vector<int> > nbrfc,
             const vector< vector<int> > p_xyz,
             const vector< int> bc,
             const double h, 
             const int rank, const int numnodes ){

    vector<double> Ap_vec;
    vector<double> U, r, p;
    double Lx, Ly, Lz, V;
    double bufferSize;
    int iterations = 0;
    double sum1=0, sum2=0; double buffer1, buffer2;
    double alpha, beta, error=10000, tol=1.0e-8;

    V = b.size();
    Lx = pow(1.0*V,1.0/3.0); Ly = Lx; Lz = Ly;

    bufferSize = 6.0 * Lx*Lx;    

    zeros1D(Ap_vec, V);
    zeros1D(U, V);
    zeros1D(r, V);
    zeros1D(p, V + bufferSize);
    
    // set initial p and r
    for(int i=0; i<V; i++){
        p[i] = b[i];
        r[i] = b[i];
    }

    iterations = 0;
    while(error>tol && iterations<2000){
//        if(iterations == 1998){cout<<"REACHED NEAR MAX ITERATIONS"<<endl;}
        // share p across processors prior to matrix times vector
        share_p(p, p_xyz, nbrfc, V, bufferSize, rank, numnodes);

        sum1=0; sum2=0;
        // Do matrix times vector multiplications
        cg_Ap(p, Ap_vec, nbr, bc, V, h);
        for(int i=0; i<V; i++){
            sum1 += r[i]*r[i];
            sum2 += p[i]*Ap_vec[i];
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&sum1, &buffer1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&sum2, &buffer2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        alpha = buffer1 / buffer2;

        sum2=0;
        for(int i=0; i<V; i++){
            U[i] = U[i] + alpha*p[i];
            r[i] = r[i] - alpha*Ap_vec[i];
            sum2 += r[i]*r[i];
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&sum2, &buffer2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        beta  = buffer2 / buffer1;
        error = buffer2;

        for(int i=0; i<V; i++){
            p[i] = r[i] + beta*p[i];
        }
        
        zeros1D(Ap_vec, V);
        iterations++;
    }

    for(int i=0; i<V; i++){
        final_U[i] = U[i]; // set final result
    }
}

// ----------------------------------------------------------------------------
// Shares the 1st input vector across processors given knowledge of the 
// neighboring processors, current rank, and information on the grid poitns 
// and their location on the faces of a given processor.
// ----------------------------------------------------------------------------
void share_p(vector<double > & p, const vector< vector<int > >& p_xyz,
             const vector< vector<int > > & nbrfc,            
             const int V, const int full_buffer_size,
             const int rank, const int numnodes){
    if(numnodes == 1){ return; } // no sharing needed for single node
    int pxp,pxm,pyp,pym,pzp,pzm;    
    int buffer_size = full_buffer_size / 6;
    // 3 = px+   4 = px-
    // 5 = py+   6 = py-
    // 7 = pz+   8 = pz-
    pxp = p_xyz[rank][3]; pxm = p_xyz[rank][4];
    pyp = p_xyz[rank][5]; pym = p_xyz[rank][6];
    pzp = p_xyz[rank][7]; pzm = p_xyz[rank][8];

    // MPI things
    MPI_Request request1, request2;

    // Declare buffers
    vector<double > sendBuffer_xp, recvBuffer_xm; // send xp
    vector<double > sendBuffer_xm, recvBuffer_xp; // send xm
    vector<double > sendBuffer_yp, recvBuffer_ym;
    vector<double > sendBuffer_ym, recvBuffer_yp;
    vector<double > sendBuffer_zp, recvBuffer_zm;
    vector<double > sendBuffer_zm, recvBuffer_zp;

    // Initialize buffers
    zeros1D(sendBuffer_xp,buffer_size); zeros1D(recvBuffer_xm,buffer_size);
    zeros1D(sendBuffer_xm,buffer_size); zeros1D(recvBuffer_xp,buffer_size);
    zeros1D(sendBuffer_yp,buffer_size); zeros1D(recvBuffer_ym,buffer_size);
    zeros1D(sendBuffer_ym,buffer_size); zeros1D(recvBuffer_yp,buffer_size);
    zeros1D(sendBuffer_zp,buffer_size); zeros1D(recvBuffer_zm,buffer_size);
    zeros1D(sendBuffer_zm,buffer_size); zeros1D(recvBuffer_zp,buffer_size);
    // Fill the send buffers from v
    MPI_Barrier(MPI_COMM_WORLD);


    // use nbrfc to set this...
    for(int i=0; i<buffer_size; i++){
        sendBuffer_xp[i] = p[ nbrfc[0][i] ];
        sendBuffer_xm[i] = p[ nbrfc[1][i] ];
        sendBuffer_yp[i] = p[ nbrfc[2][i] ];
        sendBuffer_ym[i] = p[ nbrfc[3][i] ];
        sendBuffer_zp[i] = p[ nbrfc[4][i] ];
        sendBuffer_zm[i] = p[ nbrfc[5][i] ];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards positive x (right edge x -> left edge x)
    MPI_Isend(&sendBuffer_xp[0], buffer_size, MPI_DOUBLE,
              pxp, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer_xm[0], buffer_size, MPI_DOUBLE,
              pxm, pxm, MPI_COMM_WORLD, &request1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards negative x
    MPI_Isend(&sendBuffer_xm[0], buffer_size, MPI_DOUBLE,
              pxm, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer_xp[0], buffer_size, MPI_DOUBLE,
              pxp, pxp, MPI_COMM_WORLD, &request1); 

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards positive y
    MPI_Isend(&sendBuffer_yp[0], buffer_size, MPI_DOUBLE,
              pyp, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer_ym[0], buffer_size, MPI_DOUBLE,
              pym, pym, MPI_COMM_WORLD, &request1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards negative y
    MPI_Isend(&sendBuffer_ym[0], buffer_size, MPI_DOUBLE,
              pym, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer_yp[0], buffer_size, MPI_DOUBLE,
              pyp, pyp, MPI_COMM_WORLD, &request1); 

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards positive z
    MPI_Isend(&sendBuffer_zp[0], buffer_size, MPI_DOUBLE,
              pzp, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer_zm[0], buffer_size, MPI_DOUBLE,
              pzm, pzm, MPI_COMM_WORLD, &request1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Send towards negative zs
    MPI_Isend(&sendBuffer_zm[0], buffer_size, MPI_DOUBLE,
              pzm, rank, MPI_COMM_WORLD, &request1);
    MPI_Irecv(&recvBuffer_zp[0], buffer_size, MPI_DOUBLE,
              pzp, pzp, MPI_COMM_WORLD, &request1); 

    MPI_Barrier(MPI_COMM_WORLD);

    // Loop through and assign to v buffers    
    for(int i=0; i<buffer_size; i++){
        p[V + i + buffer_size*0.0] = recvBuffer_xm[i];
        p[V + i + buffer_size*1.0] = recvBuffer_xp[i];
        p[V + i + buffer_size*2.0] = recvBuffer_ym[i];
        p[V + i + buffer_size*3.0] = recvBuffer_yp[i];
        p[V + i + buffer_size*4.0] = recvBuffer_zm[i];
        p[V + i + buffer_size*5.0] = recvBuffer_zp[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// ----------------------------------------------------------------------------
// Applies matrix times vector... the Laplacian is hard coded 
// ----------------------------------------------------------------------------
void cg_Ap(const vector<double> p, vector<double> & Ap,
           const vector< vector<float> > & nbr,
           const vector<int> & bc, const int V, const double h){
    //double a = 0.2581;    
    double diag = 6.0, offdiag = -1.0;
    diag = -6.0;
    offdiag = +1.0;

    for(int n = 0; n < V ; n++){
        Ap[n] = (diag/(h*h)) * p[n]; 
        for(int m=0 ; m < 6; m++){ // m < 7
            // CHANGE TO MINUS MINUS MINUS
            Ap[n] = Ap[n]+offdiag/(h*h) * p[ nbr[n][m] ];            
        }   
    }
    

    // apply the boundary conditions
    for (int n = 0; n < bc.size(); n++){
        Ap[ bc[n] ] = 0.0;    
    }


}



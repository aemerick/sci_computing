#include "mpi.h"
#include <vector>
#include <cmath>
using namespace std;

#include "parallel_functions.h"

// ----------------------------------------------------------------------------
// Sets the p_xyz matrix, which associates the lexical rank value assigned to
// each processor to a px,py,pz coordinates, which can be used to convert
// local (x,y,z) to global (x,y,z)
void set_p_xyz(vector< vector<int > > & p_xyz, const int N){
    int p;
    int pxp, pyp, pzp;
    int pxm, pym, pzm;
    int Nx, Ny, Nz;
    Nx = pow(1.0*N,1.0/3.0); Ny = Nx; Nz = Nx;

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
// set_bc_nbr loads the neighbor array with the lexical index for the neighbors
// of site n and determines whther site n is on the boundary or not
//
// Requires calling function to pass a predeclared and properly sized matrix 
// nbr where the neighbors will be assigned. In addition, the grid dimensions 
// are required, along with the location of the center of the atom, and the 
//  conversition to real units
void set_bc_nbr(vector< vector<double > > & nbr, vector<int > & bc,
                vector< vector<int> > & nbrfc,
                const vector<double > & gridDimensions, 
                const vector<double > & center,
                const vector< vector<int> > & p_xyz, const int rank, 
                const int V, const int GLOBAL_L, const int boundary_type){

    int Lx,Ly,Lz;
    int  x, y, z;
    int xp, yp, zp, xm, ym, zm;
    int global_x, global_y, global_z; // global x y z
    int n;
    int fs, count=0;

    double h = gridDimensions[3]; // the unit conversion for grid sizes

    Lx = gridDimensions[0]; Ly = gridDimensions[1]; Lz = gridDimensions[2];
    
    fs = Lx * Lx ; // assumes Lx=Ly=Lz

    for (int x=0; x<Lx; x++){
        for(int y=0; y<Ly; y++){
            for(int z=0; z<Lz; z++){
                // calculate the global values
                global_x = x + Lx * p_xyz[rank][0];
                global_y = y + Ly * p_xyz[rank][1];
                global_z = z + Lz * p_xyz[rank][2]; 

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
                nbr[n][6] = -2.0*h / 
                           sqrt(  (global_x-center[0])*(global_x-center[0])   +
                                  (global_y-center[1])*(global_y-center[1])   +
                                  (global_z-center[2])*(global_z-center[2])  ) ;
                                
                // if on a global boundary
                // For zero boundary conditions, fill the bc vector with
                // lexical values of global boundaries. bc vector is used
                // throughout code to fix boundaries as zero... 
                // For periodic or other, do not set (bc remains length 0)
                if(boundary_type==0 || boundary_type==2){
                    if(global_x == 0 || global_y == 0 || global_z == 0 ||
                       global_x == GLOBAL_L - 1 || 
                       global_y == GLOBAL_L - 1 ||
                       global_z == GLOBAL_L - 1    ){
                    
                        bc.push_back(n);
                    }
                } // 


                

            }   
        }
    } 

    count = 0;
    for(int i = 0; i < Lx; i++){
        for(int j=0; j< Ly; j++){
            nbrfc[0][count] = get_n( Lx-1,     j,     i, gridDimensions);
            nbrfc[1][count] = get_n(    0,     j,     i, gridDimensions);

            nbrfc[2][count] = get_n(     j, Ly-1,     i, gridDimensions);
            nbrfc[3][count] = get_n(     j,    0,     i, gridDimensions);

            nbrfc[4][count] = get_n(     j,    i,   Lz-1, gridDimensions);
            nbrfc[5][count] = get_n(     j,    i,      0, gridDimensions);
    
            count ++;
        }
    }



}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Returns the local lexical order of index n for a point x,y,z
//------------------------------------------------------------------------------
int get_n(int x, int y, int z, const vector<double > & gridDimensions){
    int Lx,Ly,Lz;
    int V;
    
    

    Lx = gridDimensions[0]; Ly = gridDimensions[1]; Lz = gridDimensions[2];
    V = Lx*Ly*Lz;

    if( x >= V || y >= V || z >= V ){
        // returns maximum of x y or z
        if(x>y && x > z){ return x;}
        else if (y>x && y>z){return y;}
        else if (z>x && z>y){return z;}
//        return  z>( (x<y)?y:x ) ? z : ( (x<y)?y:x )  ;

    }else{
        return x + Lx*(y) + Lx * Ly * (z);
    }
}

// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// Shares the 1st input vector across processors given knowledge of the 
// neighboring processors, current rank, and information on the grid poitns 
// and their location on the faces of a given processor.
// ----------------------------------------------------------------------------
void share_p(vector<double > & p, const vector< vector<int > >& p_xyz,
             const vector< vector<int > > & nbrfc,            
             const int V, const int bufferSize,
             const int rank, const int numnodes){
   // if(numnodes == 1){ return; } // no sharing needed for single node
    int pxp,pxm,pyp,pym,pzp,pzm;    
   // int buffer_size = full_buffer_size / 6.0;
    int buffer_size = bufferSize;   
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

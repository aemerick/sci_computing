//===========================================
// Author: Andrew Emerick
// Final: Evolving Maxwell's Equations in a vacuum
// 11/22/13
// 
// This program utilizes a parallelized CG solver and parallelized IRL
// function to compute the ground state energies of the Helium atom
// 
// 
//// COMPILE /home/sfw/openmpi-1.7.2/bin/mpic++ -std=c++11
// mpic++ -std=c++11 -o hydrogen.exe irl_func.cpp parallel_functions.cpp hydrogen.cpp -I /home/sfw/include -L /home/sfw/lib -lgsl -lgslcblas

//// RUN   /home/sfw/openmpi-1.7.2/bin/mpirun -np ## --hostfile hostfile _____
// mpirun -np 1 --hostfile hostfile irl_p.exe
//===========================================
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <sys/unistd.h>
#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include <vector>
#include "mpi.h"
#include <fstream>
#include <sstream>

using namespace std;
#include "parallel_functions.h"
#include "maxwell.h"

#define GLOBAL_L 32
#define CSYS 1 // using to set initial condition cases
#define boundary_type 1 // 0 = 0 boundary, 1 = periodic, 2 = Sommerfeld

int main( int argc, char * argv[]){
    double h;
    double axisLength = 8.0; // -2 to 2
    double dt = 0.0001;
    int tsteps = 10000;
    int num_dump_files = 10; 
    h = axisLength/(1.0*GLOBAL_L - 1.0);

    int rc = MPI_Init(&argc,&argv);
    if (rc!=MPI_SUCCESS){
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD,rc);
    }

    int rank, numnodes;
    double buffer;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&numnodes);

    double startTime, endTime;

// ---------- Timing ----------
    if (rank == 0){
        printf("# Starting Timing\n");
        startTime = MPI_Wtime();
    }
// ---------- Timing -----------

    int global_Lx, global_Ly, global_Lz;
    int global_V;
    vector<double > center;
    vector< vector<int > > p_xyz;

    global_Lx = GLOBAL_L; global_Ly = global_Lx; global_Lz = global_Lx;
    global_V  = global_Lx * global_Ly * global_Lz;

    // set the center coordinates (in 'code' coordinates)
    zeros1D(center,3);
    center[0] = ((global_Lx -1.0)/ 2.0) + 0.5*h;
    center[1] = ((global_Ly -1.0)/ 2.0) + 0.5*h;
    center[2] = ((global_Lz -1.0)/ 2.0) + 0.5*h;

    // fills the nearest neighbor processor matrix
    if(rank==0){printf("# Setting p_xyz\n");}
    zeros2D(p_xyz,numnodes,9);

    set_p_xyz(p_xyz,numnodes);

    if(rank==0){printf("# Done setting p_xyz\n");}

    MPI_Barrier(MPI_COMM_WORLD);
// Now for more local things-------------------------------------------------------

    int Lx, Ly, Lz, V;
    vector< vector<double > > nbr;
    vector< vector<int    > > nbrfc;
    vector<int> bc;
    vector<double> gridVals;
    int global_x,global_y,global_z;
    

    Lx = global_Lx / (pow(1.0*numnodes,1.0/3.0));
    Ly = global_Ly / (pow(1.0*numnodes,1.0/3.0));
    Lz = global_Lz / (pow(1.0*numnodes,1.0/3.0));
    V = Lx*Ly*Lz;

    zeros1D(gridVals,4);
    gridVals[0] = Lx; gridVals[1] = Ly; gridVals[2] = Lz; gridVals[3] = h;

    // 
    zeros2D(nbr,V,7);
    zeros2D(nbrfc,6,Lx*Lx);

    // Call nearest neighbor function. Sets boundaries and also faces
    if(rank==0){printf("# Setting grid point relationships\n");}    


    set_bc_nbr(nbr,bc,nbrfc,gridVals,center,p_xyz,rank,V,GLOBAL_L,
               boundary_type);
    
    if(rank==0){printf("# Done setting grid point relationships\n");}

    MPI_Barrier(MPI_COMM_WORLD);
//--- Now for the fun ---------------------------------------------------------

    vector< vector<double> > E;
    vector< vector<double> > A;
    vector< vector<double> > B;
    vector<double>  Benergy;
    vector< vector<double> > A_an;
    int bufferSize = Lx*Lx;
    int ndim = 3.0; // number of dimensions... always 3 really    

    // variables used for summation
    double t, sum1,sum2,sum3,sum4,sum5,sum6;
    double EE = 0, BB = 0, BBlap = 0, UU = 0, AA = 0, AAan = 0;

    int n;

    // write out variables
    ofstream myfile, allFile;
    int write_out=0;

    // reserve space for E,B,A and associated vector fields
    zeros2D(E,ndim,V);
    zeros2D(B,ndim,V);
    zeros1D(Benergy,V);
    zeros2D(A,ndim,V + 6.0*bufferSize);
    zeros2D(A_an,ndim,V);
    
    // set the initial condition
    if(rank==0){printf("# Setting initial conditions\n");}
    initial_conditions(A,E,center,p_xyz,gridVals,rank);
    if(rank==0){printf("# Done Setting initial conditions\n");}


    t = 0.0;
    if(rank==0){ // sets filename for energy vs. time output
        std::ostringstream fileNameStream("");
        fileNameStream<<"global_values_np"<<numnodes<<".dat";
        std::string fileName = fileNameStream.str();
        allFile.open(fileName.c_str());
    }

    // iterate over the designated number of time steps
    for(int tt = 0; tt<tsteps; tt++){

        // trigger the write out code at set intervals
        if(tt == 0 || tt % (tsteps/num_dump_files) ==0){
            write_out = 1;
        }else{write_out=0;}

        // sets filename and opens dump file (each processor does this)
        if(write_out==1){ 
                std::ostringstream fileNameStream("time_");
                fileNameStream << setfill('0')
                << setiosflags(ios::fixed)<<
                setprecision(2)<<setw(3)<< t << "-p" <<setw(1)<<rank<<".dat";
                std::string fileName = fileNameStream.str();
                myfile.open(fileName.c_str());
        }


        // Calculates the analytical solution to A at time t
        MPI_Barrier(MPI_COMM_WORLD);
        A_anal(A_an,t,center,p_xyz,gridVals,rank,dt);

        // Calculates B (using the curl) and Benergy (using laplacian)
        // given the vector potential A
        MPI_Barrier(MPI_COMM_WORLD);
        calculate_B(B,Benergy,A,h,nbr,nbrfc,bc,p_xyz,rank,numnodes,bufferSize);
        MPI_Barrier(MPI_COMM_WORLD);
        
        // loop over all grid points and perform dot product. 
        // sum to calculate global energy densities
        EE = 0; BB = 0; BBlap = 0; UU = 0; AA = 0; AAan = 0;
        for(int n = 0; n < V; n++){
            for(int i = 0; i < ndim; i++){
                EE += abs(E[i][n]*E[i][n]); // electric field
                BB += abs(B[i][n]*B[i][n]); // magetic field using curl                
                AA += abs(A[i][n]*A[i][n]); // vector potential
                AAan += abs(A_an[i][n]*A_an[i][n]); // analytical vector pot.
            }
                BBlap += abs(Benergy[n]); // magnetic field using laplacian
        }
        UU += (EE + BB) / 2.0; // add to get total energy density
        
        // Do series of MPI all sum's to get global result
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&EE,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        EE = buffer;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&BB,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        BB = buffer;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&UU,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        UU = buffer;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&BBlap,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        BBlap = buffer;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&AAan,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        AAan = buffer;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&AA,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        AA = buffer;

        // Multiply by grid spacing cubed to get energies from the densities
        EE = EE*h*h*h; BB = BB*h*h*h; BBlap = BBlap*h*h*h;
        AA = AA*h*h*h; UU = UU*h*h*h; AAan  = AAan *h*h*h;


        if(rank==0){

            // make a print to screen of progress if this is a dump iteration
            if(write_out==1){
                printf("# E*E at time %f is %e\n", t, EE);
                printf("# B*B curl at time %f is %e\n", t, BB);
                printf("# B*B laplacian at time %f is %e\n", t, BBlap);
                printf("# A*A at time %f is %e\n", t, AA);
                printf("# A*A analytical at time %f is %e\n", t, AAan);
                printf("# The total energy at time %f is %e\n", t, UU);
                printf("# The total energy at time %f is %e\n", t, (EE+BBlap)*0.5);
           }
            
            // output global energies to file every time step
            allFile<<setw(8)<<t<<setw(15)<<EE  <<setw(15)<<BB   <<setw(15)<<AA<<
                                 setw(15)<<AAan<<setw(15)<<BBlap<<setw(15)<<UU<<
                                 setw(15)<<0.5*(EE+BBlap)<<endl;

        }

        MPI_Barrier(MPI_COMM_WORLD);


        if(write_out==1){ // make a write out of entire simulation volume
            for(int z = 0; z<Lz; z++){
                for(int y = 0; y<Ly; y++){
                    for(int x = 0; x<Lx; x++){
                        global_x = x + Lx*p_xyz[rank][0];
                        global_y = y + Ly*p_xyz[rank][1];
                        global_z = z + Lz*p_xyz[rank][2];

                        n=get_n(x,y,z,gridVals);
        
                        // Get energy densities at each grid point
                        sum1 = E[0][n]*E[0][n] + E[1][n]*E[1][n] + E[2][n]*E[2][n];
                        sum2 = B[0][n]*B[0][n] + B[1][n]*B[1][n] + B[2][n]*B[2][n];
                        sum3 = A[0][n]*A[0][n] + A[1][n]*A[1][n] + A[2][n]*A[2][n];
                        sum4 = A_an[0][n]*A_an[0][n] + A_an[1][n]*A_an[1][n] + A_an[2][n]*A_an[2][n];

                        // convert to energy
                        sum1 = h*h*h*sum1;
                        sum2 = h*h*h*sum2;
                        sum3 = h*h*h*sum3;
                        sum4 = h*h*h*sum4;
                        myfile<<setw(5)<<global_x<<setw(5)<<global_y<<setw(5)<<global_z<<
                                setw(15)<<sum1<<setw(15)<<sum2<<setw(15)<<(sum1+sum2)*0.5<<setw(15)<<
                                sum3<<setw(15)<<sum4<<endl;
 
                    }
                }
            }
            myfile.close(); // close output for each processor
            
        }

        // make sure everyone is finished writing to file
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Evolve the system by dt using the iterated Crank Nicholson scheme
        cn_evolve(A,E,dt,h, center, gridVals,
                  nbr,nbrfc,p_xyz,bc,rank,numnodes,bufferSize);

        MPI_Barrier(MPI_COMM_WORLD);
        t += dt; // advance time
    }

    allFile.close(); // close global energy vs. time file

    MPI_Finalize();
    return 0;
}

// -----------------------------------------------------------------------------
// Given A, E and information on the simulation volume, this function supplies
// initial conditions on A and E:
void initial_conditions(vector< vector<double> > & A,
                        vector< vector<double> > & E,
                        const vector<double> & rloc,
                        const vector< vector<int> > & p_xyz,
                        const vector<double> & gridVals,
                        const int rank){
    int V;
    int Lx, Ly, Lz, n;
    int global_x, global_y, global_z;
    int ndim = A.size();
    double h;
    double r, theta, phi, exp_term;
    double y_loc, rsin;

    Lx = gridVals[0]; Ly = gridVals[1] ; Lz = gridVals[2];
    h  = gridVals[3];
    V = Lx*Ly*Lz;

    // Loop over all x,y,z,... uses get_n to get the lexical order index
    for(int x=0; x<Lx; x++){
        for(int y=0; y<Ly; y++){
            for(int z=0; z<Lz; z++){
                global_x = x + Lx*p_xyz[rank][0];
                global_y = y + Ly*p_xyz[rank][1];
                global_z = z + Lz*p_xyz[rank][2];

                r = h*sqrt((rloc[0]-global_x)*(rloc[0]-global_x) +
                           (rloc[1]-global_y)*(rloc[1]-global_y) +
                           (rloc[2]-global_z)*(rloc[2]-global_z) );

                n = get_n(x,y,z,gridVals);

                // (NAMES ARE REVERSED... NEED TO FIX...)
                // used to convert from spherical to cartesian coordinates
                theta = atan2( (global_y-rloc[1]), (1.0*(global_x-rloc[0])) );
                phi   = sqrt( (global_y-rloc[1]) * (global_y-rloc[1]) +
                              (global_x-rloc[0]) * (global_x-rloc[0]) );
                phi   = atan2(phi,1.0*(global_z - rloc[2]));

                exp_term = exp(-r*r);


                // set the initial A and E
                for(int i = 0; i < ndim; i++){
                    A[i][n] = 0.0;   // A is initially zero everywhere              
                }

                if(CSYS==0){
                    E[0][n] =  8.0*r*sin(theta)*cos(theta)*cos(phi)*exp_term;
                    E[1][n] =  8.0*r*sin(theta)*sin(theta)*cos(phi)*exp_term;
                    E[2][n] = -8.0*r*sin(theta)*sin(phi)*exp_term;
                }else if(CSYS==1){ // set conditions given in problem set...
                                   // using conversion from spherical to 
                                   // cartesian vector field....
                    E[0][n] = -8.0*r*sin(theta)*sin(phi)*exp_term;
                    E[1][n] =  8.0*r*cos(theta)*sin(phi)*exp_term;
                    E[2][n] =  0.0;
    
                }

            }
        }
    }  




}
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Evolves A and E using the twice iterated Crank Nicholson scheme
// Requires knowledge of the simulation volume locally and globally
void cn_evolve(vector< vector<double> > & A, vector< vector<double> > & E,
               const double dt, const double dx, const vector<double> & center,
               const vector<double> & gridVals,
               const vector< vector<double> > & nbr,
               const vector< vector<int> > & nbrfc,
               const vector< vector<int> > & p_xyz, 
               const vector<int> & bc,
               const int rank, const int numnodes, const int bufferSize){
// NOTE: A_prev, E_prev and A_old, E_old vectors are currently there to
//       test out things with the Sommerfeld boundary conditions... they
//       are superflous otherwise.......................................

    vector< vector<double> > A_prev;
    vector< vector<double> > E_prev;
    vector<double> E_new, E_old;
    vector<double> A_new, A_old;
    vector<double> lapA;
    int ndim = A.size(), V = E[0].size();
    double h = gridVals[3];

    zeros2D(E_prev,3,V); zeros2D(A_prev,3,V);

    zeros1D(E_new,V);
    zeros1D(A_new,A[0].size());
    zeros1D(E_old,V);
    zeros1D(A_old,V);
    zeros1D(lapA ,V);

    for(int i = 0 ; i < ndim; i++){ // Do once for each dimension
     
        // make some copies 
        for(int j = 0; j < V; j++){
            E_new[j] = E[i][j];
            E_old[j] = E_new[j];
            E_prev[i][j] = E[i][j];
            A_prev[i][j] = A[i][j];
            A_old[j] = A[i][j];
        }
        for(int j = 0; j < A[0].size(); j++){
            A_new[j] = A[i][j];
        }


        for(int iter=0; iter<2; iter++){ // Do two CN iterations

            // need to share across faces 
            MPI_Barrier(MPI_COMM_WORLD);
            share_p(A_new, p_xyz, nbrfc, V, bufferSize, rank, numnodes);
            MPI_Barrier(MPI_COMM_WORLD);

            // Calculates the laplacian of A used to update E
            laplacian(A_new, lapA, nbr, bc, V, h);
        
            // Iterate twice through the Crank Nicholson scheme... done
            // simultaneously with both A and E
            for(int j = 0; j < V; j++){
                A_new[j] = A_old[j] - E_new[j] * dt;   // A tilde n+1
                E_new[j] = -dt*lapA[j] + E_old[j];     // E tilde n+1
  
                A_new[j] = 0.5*(A_new[j] + A_old[j]);  // A bar n+1/2
                E_new[j] = 0.5*(E_new[j] + E_old[j]);  // E bar n+1/2
            }
        }

        // Share across processors for the final update
        MPI_Barrier(MPI_COMM_WORLD);
        share_p(A_new, p_xyz, nbrfc, V, bufferSize, rank, numnodes);
        MPI_Barrier(MPI_COMM_WORLD);

        // Take the Laplacian one last time
        laplacian(A_new, lapA, nbr, bc, V, h);

        // Do the final update of A and E
        for(int j = 0; j < V; j++){
            A_new[j] = A_old[j] - E_new[j] * dt;     // A n+1
            E_new[j] = -dt * lapA[j] + E_old[j];     // E n+1
        }

       // if(boundary_type==2){
                   
         //   sommerfeld(A_new,A_old,bc,nbr);
           // sommerfeld(E_new,E_old,bc,nbr);
              
        //}

        // Officially update A and E
        for(int j = 0; j < V; j++){
            A[i][j] = A_new[j];
            E[i][j] = E_new[j];
        }

    } // End dimension loop    


    if(boundary_type==0){ // zero boundary conditions
        for(int i=0; i<bc.size(); i++){
            for(int j=0; j<3; j++){
                A[j][ bc[i] ] = 0.0; E[j][ bc[i] ] = 0.0;
            }
        }
    }else if(boundary_type==2){    // Do stuff wit sommerfeld if enabled... not yet working....
        sommerfeld(A,A_prev,bc,nbr);
        sommerfeld(E,E_prev,bc,nbr);
    }


}

// -----------------------------------------------------------------------------
// The goal of this function is to be able to take in a vector field given its 
// current (n) and previous (n-1) values and employs the Sommerfeld radiation
// boundary conditions... this is not too easy, and may ultimately require
// knowledge of the (n-2) vector field... if true, this would require some
// significant code rewriting.... maybe could have the CN do two time steps
// at a time of size dt' = dt*0.5, and then use these to get n, n-1, n-2...
// ... hmmm..... may also need to know about the grid points two spacings away
// from the boundary, instead of just one away... this would require
// doubling the size of the nbr matrix and a little bit of thinking
void sommerfeld(vector< vector<double> >  & v, const vector< vector<double> > & v_old,
                const vector<int> & bc, 
                const vector< vector<double> > & nbr){
    int n;
    int xp, xm, yp, ym, zp, zm, adjBin;
    int ndim = v_old.size();
    int V = v_old[0].size();

    for(int i = 0; i < bc.size(); i++){
        n = bc[i];
        
        xp = nbr[n][0]; xm = nbr[n][1]; 
        yp = nbr[n][2]; ym = nbr[n][3];
        zp = nbr[n][4]; zm = nbr[n][5];

        // diffuse things in the right direction
        if(xp >= V){
            // diffuse out to the right... so use xm
            v[0][n] = v_old[0][xm];
        }else if(xm >=V){
            v[0][n] = v_old[0][xp];

        }

        if(yp >= V){
            v[1][n] = v_old[1][ym];
        }else if (ym >=V){
            v[1][n] = v_old[1][yp];

        }

        if(zp >= V){
            v[2][n] = v_old[2][zm];
        }else if (zm >=V){
            v[2][n] = v_old[2][zp];
        }
         
    }



}

// -----------------------------------------------------------------------------
// Calculates the laplacian of a given vector. Result is given in Av
//
void laplacian(const vector<double> v, vector<double> & Av,
        const vector< vector<double> > & nbr,
        const vector<int> & bc, const int V, const double h){

    double diag,offdiag;

    diag =    -6.0/(h*h);
    offdiag =  1.0/(h*h);

    for(int n = 0; n < V ; n++){
        Av[n] = diag * v[n]; 
        for(int m=0 ; m < 6; m++){ // m < 7
           Av[n] = Av[n] + offdiag * v[ nbr[n][m] ];            
        }   
    }

   // apply the zero boundary conditions (if set)
    if(boundary_type==0){
        for (int n = 0; n < bc.size(); n++){
            Av[ bc[n] ] = 0.0;    
        }
    }
}
// -----------------------------------------------------------------------------
// Function calculates B from the vector potential. B is calculated using 
// a discretized curl. In addition, Benergy vector is made, which should contain
// B*B (B dot B) at each grid point. This is performed separately from the 
// curl method, using the laplacian.
void calculate_B(vector< vector<double> > & B,
                 vector<double> & Benergy,
                 vector< vector<double> > & A,
                 const double h,
                 const vector< vector<double> > & nbr,
                 const vector< vector<int> > & nbrfc,
                 const vector<int> & bc,
                 const vector< vector<int> > & p_xyz,
                 const int rank, const int numnodes, const int buffer_size){

    int V = B[0].size();
    int xp, xm, yp, ym, zp, zm;
    vector< vector<double> > B_old;
    vector<double> A_temp;
    vector<double> share_temp0;
    vector<double> share_temp1;
    vector<double> share_temp2;

    zeros2D(B_old,3,B[0].size());
    zeros1D(A_temp,A[0].size());
    zeros1D(share_temp0,A[0].size());
    zeros1D(share_temp1,A[0].size());
    zeros1D(share_temp2,A[0].size());

    for(int i = 0; i<A[0].size(); i++){
        share_temp0[i] = A[0][i];
        share_temp1[i] = A[1][i];
        share_temp2[i] = A[2][i];
    }
    for(int i = 0; i<V; i++){
        for(int j = 0; j<3; j++){
            B_old[j][i] = B[j][i];
        }
    }


    //Need to share A_x , A_y, and A_z across faces in order to take laplacians
    share_p(share_temp0,p_xyz,nbrfc,V,buffer_size,rank,numnodes);
    share_p(share_temp1,p_xyz,nbrfc,V,buffer_size,rank,numnodes);
    share_p(share_temp2,p_xyz,nbrfc,V,buffer_size,rank,numnodes);

    
    for(int i = 0; i<A[0].size(); i++){
        A[0][i] = share_temp0[i];
        A[1][i] = share_temp2[i];
        A[2][i] = share_temp2[i];
    }



    for(int n = 0; n < V; n++){    // the discretized curl
        xp = nbr[n][0];
        xm = nbr[n][1];
        yp = nbr[n][2];
        ym = nbr[n][3];
        zp = nbr[n][4];
        zm = nbr[n][5];
    
        B[0][n] = (A[2][yp] - A[2][ym]) -
                  (A[1][zp] - A[1][zm]) ;
        B[1][n] = (A[0][zp] - A[0][zm]) -
                  (A[2][xp] - A[2][xm]) ;
        
        B[2][n] = (A[1][xp] - A[1][xm]) -
                  (A[0][yp] - A[0][ym]) ;

        B[0][n] = B[0][n] / (2.0*h);
        B[1][n] = B[1][n] / (2.0*h);
        B[2][n] = B[2][n] / (2.0*h);
    }

    for(int n = 0; n<A[0].size();n++){ //used to take laplacian
        A_temp[n] = -(A[0][n]*A[0][n] + A[1][n]*A[1][n] + A[2][n]*A[2][n]);
    }

    laplacian(A_temp,Benergy,nbr,bc,V,h);

    if(boundary_type==0){
        for(int i = 0; i < bc.size(); i++){
            for(int j=0; j<3; j++){
                B[j][ bc[i] ] = 0.0;
            }
            Benergy[ bc[i] ] = 0.0;
        }
    }else if(boundary_type==2){ // do stuff with the not-yet-working sommerfeld
        sommerfeld(B,B_old,bc,nbr);
    }
}

// -----------------------------------------------------------------------------
// This function should be calculating the analytical solution to A at 
// any given time t. Currently blows up severly (worse at large r)... Not sure
// why this is the case....
void A_anal (vector< vector<double> > & A, const double & t,
                        const vector<double> & rloc,
                        const vector< vector<int> > & p_xyz,
                        const vector<double> & gridVals,
                        const int rank, const double dt){
    int V;
    int Lx, Ly, Lz, n;
    int global_x, global_y, global_z;
    int ndim = A.size();
    double h;
    double r;
    double phi, theta, exp_term, u, v;

    Lx = gridVals[0]; Ly = gridVals[1] ; Lz = gridVals[2];
    h  = gridVals[3];
    V = Lx*Ly*Lz;

    // Loop over all x,y,z,... uses get_n to get the lexical order index
    for(int x=0; x<Lx; x++){
        for(int y=0; y<Ly; y++){
            for(int z=0; z<Lz; z++){
                n = get_n(x,y,z,gridVals);

                global_x = x + Lx*p_xyz[rank][0];
                global_y = y + Ly*p_xyz[rank][1];
                global_z = z + Lz*p_xyz[rank][2];

                r =   sqrt((rloc[0]-global_x)*(rloc[0]-global_x) +
                             (rloc[1]-global_y)*(rloc[1]-global_y) +
                             (rloc[2]-global_z)*(rloc[2]-global_z) );
                r=h*r;
                u = t + r;
                v = t - r;

                theta = atan2( (global_y-rloc[1]) , (1.0*(global_x-rloc[0])) );
                phi   = sqrt( (global_y-rloc[1]) * (global_y-rloc[1]) +
                              (global_x-rloc[0]) * (global_x-rloc[0]) );
                phi   = atan2( phi, 1.0*(global_z - rloc[2]) );
  

                exp_term = (exp(-v*v) - exp(-u*u))/(r*r) - 
                           2.0*( (v*exp(-v*v) + u*exp(u*u))/r);

                if(CSYS==0){
                    A[0][n] = sin(theta)*cos(theta)*cos(phi)*exp_term;
                    A[1][n] = sin(theta)*sin(theta)*cos(phi)*exp_term;
                    A[2][n] = sin(theta)*sin(phi)*exp_term;
                } else if(CSYS==1){
                    A[0][n] = -sin(theta)*sin(phi)*exp_term;
                    A[1][n] =  cos(theta)*sin(phi)*exp_term;
                    A[2][n] = 0.0;
                }

            }
        }
    }  




}

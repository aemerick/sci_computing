//===========================================
// Author: Andrew Emerick
// Homework 2: Hydrogen atom
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
#define _USE_MATH_DEFINES // for pi in math.h
#include <math.h>

#include <iostream>
#include <sys/unistd.h>
#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include <vector>


#include "mpi.h"


using namespace std;
// my own headers
//#include "parallel_functions.h"
#include "parallel_functions.h"     // to set up the volume and sharing 
#include "cg_func.h"                // CG solver
#include "irl_func.h"               // IRL solver
#include "helium.h"               // 

/* Specifies the number of global grid points in each dimension */
#define GLOBAL_L 40

int main( int argc , char * argv[]){
    double h;
    int dft_iterations = 10;
    double axisLength = 8.0;
    h = axisLength/(1.0*GLOBAL_L -1.0);
   

// initialize MPI & processors
    int rc = MPI_Init( &argc, &argv);     
    if ( rc!= MPI_SUCCESS) {
        cout <<"Error starting MPI progam. Terminating."<<endl;
        MPI_Abort(MPI_COMM_WORLD,rc);
    }
    
    int rank, numnodes;
    double double_recv;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &numnodes);
    double startTime, endTime;

// --------- Timing ---------------------
	if (rank == 0){ 
		 cout<<"# Starting Timing"<<endl;
		 startTime = MPI_Wtime();
	}
// --------- End Timing ------------------

    // Declare global variables
    int    global_Lx, global_Ly, global_Lz;
    double global_V;
    vector<double > rloc;
    vector< vector<int > > p_xyz;

    global_Lx = GLOBAL_L; global_Ly = global_Lx; global_Lz = global_Lx;    
    global_V  = global_Lx * global_Ly * global_Lz;
    
    // Set the center coordinates (arbitrary center)
    zeros1D(rloc,3);
    rloc[0] = global_Lx/2.0 - 3.2;
    rloc[1] = global_Ly/2.0 - 1.6; 
    rloc[2] = global_Lz/2.0 + 2.7;


    // fills matrix associating the lexical rank to a px,py,pz grid of
    // processors. Used to convert local to global coordinates for 
    // the set_nbr_bc function. Set to be even 3D grid
    if(rank==0){cout<<"#Setting p_xyz"<<endl;}
    zeros2D(p_xyz,numnodes,9); 

    set_p_xyz(p_xyz,numnodes);

    if(rank==0){cout<<"#Done setting p_xyz"<<endl;}
    MPI_Barrier( MPI_COMM_WORLD);
// ------------------------------------------------------------------------
    int Lx, Ly, Lz, V;
    vector< vector<float > > nbr; // contains references among local grid points
    vector< vector<int > > nbrfc; // contains references to grid points on faces 
    vector<int> bc;               // contains references to global boundary pnts
    vector<double > gridVals;     // contains grid parameters (Lx,Ly,Lz,a)


    Lx = global_Lx / (pow(1.0*numnodes,1.0/3.0));
    Ly = global_Ly / (pow(1.0*numnodes,1.0/3.0)); 
    Lz = global_Lz / (pow(1.0*numnodes,1.0/3.0)); 
    V  = Lx*Ly*Lz;
    zeros1D(gridVals,4);

    gridVals[0] = Lx; gridVals[1] = Ly; gridVals[2] = Lz; gridVals[3] = h;


    // Appropriately size all vectors and matrices
    // 0=xp 1=xm 2=yp 3=ym 4=zp 5=zm 6=local 1/r term
    zeros2D(nbr,V,7);
    zeros2D(nbrfc, 6, Lx*Lx); 

    // Call nearest neighbor function to set nbr and nbrfc
    // also marks elements in v that correspond to a global boundary in bc
    if(rank==0){cout<<"#Setting grid point relationships"<<endl;}
   
    set_bc_nbr(nbr,bc,nbrfc,gridVals,rloc,p_xyz,rank,V,GLOBAL_L);

    if(rank==0){cout<<"#Done setting grid point relationships"<<endl;}

    MPI_Barrier( MPI_COMM_WORLD);
//-----------------------------------------------------------------------------
    // Now worrying about local things
    vector< vector<double> > eigenvec;
    vector< double> eigenval;
    int edet, bufferSize;
    bufferSize = 6.0* Lx*Lx;

    double Etot, T, Ex, Eh, Eext, sum_n=0;
    double buffer=0;
    vector<double> psi;
    vector<double > Vtot;
    vector<double > Vext;
    vector<double > Vx, Vh;
    vector<double > ncomp, ncomptot;
    vector<double > b,n;


    // Initialize psi.. includes volume and buffer for info share 
    zeros1D(psi, V + bufferSize);

    // Initialize Density vector
    zeros1D(n,V);

    // initiallize b vector for cg solver
    zeros1D(ncomp,V);
    zeros1D(ncomptot,V);
    zeros1D(b,V);

    // initialize vectors for IRL
    edet = 1; // Just determine the ground state..
    zeros2D(eigenvec, V, edet);
    zeros1D(eigenval,    edet);

    // Initialize potential vectors
    zeros1D(Vext,V);
    zeros1D(Vx  ,V);
    zeros1D(Vh  ,V);
    zeros1D(Vtot,V);

    // fills v_external and therfore initial guess potential in nbr    
    fill_vext_ncomp(Vext,ncomp,ncomptot,gridVals,rloc,p_xyz,rank);
    for(int i=0; i<V; i++){
        Vtot[i]   = Vext[i];
    }


    for(int nl = 0; nl<dft_iterations;nl++){
        // set potential in the nbr matrix... used in IRL solver
        for(int i=0; i<V; i++){
            nbr[i][6] = Vtot[i];
        }

        // Find the first edet eigenvalues and eigenvectors... aka Psi and E
        if(rank==0){printf("Iteration %d: Beginning IRL Algorithm\n",nl);}
        MPI_Barrier(MPI_COMM_WORLD);
        irl_func(eigenvec, eigenval, edet, gridVals, nbr, nbrfc, p_xyz, bc, rank, numnodes);

        if(rank==0){printf("Iteration %d: Completed IRL. Obtained eigenvalue %e \n", nl, eigenval[0]);}

        // Loop extracts psi from the ground state eigenvector
        for(int i=0; i<V; i++){
          // psi[i] = eigenvec[i][0];
           psi[i] = eigenvec[i][0] / (pow(1.0*h,3.0/2.0));
           //psi[i] = eigenvec[i][0]/(pow(1.0*a,3.0/2.0)); // following MK
        }

        // Computes the number density of the state
        // Computes the integrated number density (should be 2)
        // Computes the exchange potential (Vx)
        sum_n = 0;
        for(int i =0; i<V; i++){
            n[i] = 2.0 * psi[i]*psi[i]; // following MK
            sum_n += n[i];
            //Vx[i] =  -1.0*pow((3.0/sqrt(M_PI)),1.0/3.0) * pow(n[i],1.0/3.0);
            Vx[i] =  -1.0*pow(3.0/M_PI,1.0/3.0) * pow(n[i],1.0/3.0);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&sum_n,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        sum_n = buffer; // communicate with all processors


        if(rank==0){printf("Iteration %d: sum of n is %e\n",nl,sum_n);}

        // Calculate b to be used in CG solver 
        for(int i=0; i<V; i++){
//            b[i] = 4.0*M_PI*(n[i] + ncomp[i]); // negative sign?
            b[i] = -4.0*M_PI*(n[i] + ncomp[i]); // following MK
        }
        sum_n = 0;
        for(int i = 0 ; i< V; i++){
            sum_n += b[i];
        }

        for(int i=0; i<bc.size(); i++){ // enforced boundary conditions
            b[ bc[i] ] = 0.0; 
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&sum_n,&buffer,1, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        if(rank==0){cout<<"sum b "<<buffer<<endl;}



        // Solve and place in Vh
        if(rank==0){printf("Iteration %d: Beginning CG function\n",nl);}
        MPI_Barrier(MPI_COMM_WORLD);
        cg_func(b,Vh,nbr,nbrfc,p_xyz,bc,h,rank,numnodes);

        if(rank==0){printf("Iteration %d: Completed CG Solver. Obtained Vh\n",nl);}

        // Calculate Vh
        for(int i=0; i<V; i++){
            Vh[i] = (Vh[i] - ncomptot[i]);
        }
        sum_n = 0;
        for(int i = 0 ; i< V; i++){
            sum_n += ncomptot[i];
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&sum_n,&buffer,1, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        if(rank==0){cout<<"sum ncomptot "<<buffer<<endl;}

        sum_n = 0;
        for(int i = 0 ; i< V; i++){
            sum_n += Vh[i];
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&sum_n,&buffer,1, MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        if(rank==0){cout<<"sum V "<<buffer<<endl;}
        

        // Calculates Vtotal
        for(int i=0; i<V; i++){
            Vtot[i] = Vx[i] + Vext[i] + Vh[i];
        }

        // Calculate energies

        T = 0; Eext = 0; Ex = 0; Eh = 0; 
        for(int i=0; i<V; i++){
//            Eext += -n[i]*Vext[i];
  //          Ex   += 3.0/4.0 * pow(3.0/M_PI,1.0/3.0) * pow(n[i],4.0/3.0);
    //        Eh   += - 0.5 * n[i]*Vh[i];
          

             // following MK
         //   Eext += (n[i]*Vext[i])*(a*a*a);
          //  Ex   += (-3.0/4.0 * pow(3.0/M_PI,1.0/3.0) * pow(n[i],4.0/3.0))*(a*a*a);
          //  Eh   += 0.5 * n[i]*Vh[i]*(a*a*a);

            Eext += (n[i]*Vext[i]);//*(a*a*a);
            Ex   += (-3.0/4.0 * pow(3.0/M_PI,1.0/3.0) * pow(n[i],4.0/3.0));//*(a*a*a);
            Eh   += 0.5 * n[i]*Vh[i];//*(a*a*a);
        }

        // Calculates the kinetic energy
        T = calculate_T(psi, p_xyz, nbr, nbrfc, bc, bufferSize, h, rank, numnodes);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&Eext,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        Eext = buffer;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&Ex,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        Ex = buffer;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&Eh,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        Eh = buffer;

        T=T*h*h*h;  Eext=Eext*h*h*h; Ex = Ex*h*h*h; Eh = Eh*h*h*h;
        // Sum for the total energy
        Etot = T + Eext + Ex + Eh;

        if(rank == 0){
            cout << "T "<< T<<endl;
            cout << "Eext "<<Eext<<endl;
            cout << "Ex "<<Ex<<endl;
            cout << "Eh "<<Eh<<endl;
            cout << "Etot "<<Etot<<endl;
        }    
        MPI_Barrier(MPI_COMM_WORLD);
    }


    double time_taken;
    if(rank ==0 ){
        endTime = MPI_Wtime();
        time_taken = endTime - startTime;
        printf("Finished in %e s (%e min)",time_taken,time_taken/60.0);
    }

    MPI_Finalize();
    return 0;
}

// -----------------------------------------------------------------------------
// Function calculates the initial external potential used as the initial 
// conditions for the total potential. In addition, calculates ncomp and
// ncomptot, or the compensating vectors used to compensate for the 
// implicit zero boundary conditions employed.
// -----------------------------------------------------------------------------
void fill_vext_ncomp(vector<double> & Vext,
               vector<double> & ncomp,
               vector<double> & ncomptot,
               const vector<double> & gridVals, 
               const vector<double> & rloc,
               const vector< vector<int> > & p_xyz, const int rank){

    int V;
    int Lx, Ly, Lz, n;
    int global_x, global_y, global_z;
    double h;
    double r;

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

                // distance from center
              /*  r = sqrt((rloc[0]-global_x)*(rloc[0]-global_x) +
                         (rloc[1]-global_y)*(rloc[1]-global_y) +
                         (rloc[2]-global_z)*(rloc[2]-global_z) );*/

                r = h*sqrt((rloc[0]-global_x)*(rloc[0]-global_x) +
                         (rloc[1]-global_y)*(rloc[1]-global_y) +
                         (rloc[2]-global_z)*(rloc[2]-global_z) );

                n = get_n(x,y,z,gridVals);
            //    Vext[n] = - 2.0  * a /r ; // external potential

            //    ncomp[n] = exp(-(r/a)*(r/a)/2.0);       
    
                //ncomptot[n] = -(2.0/(r/a))*erf((r/a)/sqrt(2.0));


/*
                Vext[n] = - 2.0 *a /(r) ; // external potential

                ncomp[n] = exp(-(r/a)*(r/a)/2.0);       
    
                ncomptot[n] = -(2.0/(r/a))*erf((r/a)/sqrt(2.0));*/

                Vext[n] = - 2.0 /r ; // external potential

                ncomp[n] = exp(-(r)*(r)/2.0);       
    
                ncomptot[n] = -(2.0/(r))*erf((r)/sqrt(2.0));

            }
        }
    }  

    // Use to divide ncomp by its sum
    double sum=0, buffer=0;
    for(int i=0;i<V;i++){
        sum += ncomp[i]; // following MK
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&sum,&buffer,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    sum = buffer; // following MK

    for(int i = 0; i<V;i++){
        ncomp[i] = -2.0*(ncomp[i]/sum)/(h*h*h);// /(a*a*a) ; // following MK
    }

        


}

// -----------------------------------------------------------------------------
// Calculates the total kinetic energy
// -----------------------------------------------------------------------------
double calculate_T(vector<double > & psi, const vector< vector<int > > & p_xyz,
                   const vector< vector<float> > & nbr,                   
                   const vector< vector<int > > & nbrfc, 
                   const vector<int> & bc,
                   const int full_buffer_size, const double h, const int rank, 
                   const int numnodes){
    int V = nbr.size();
    double sum, buffer;
    
    vector<double> Apsi; // store matrix times vector
    zeros1D(Apsi,V);

    // Share psi across processors
    share_p(psi, p_xyz, nbrfc, V, full_buffer_size, rank, numnodes);

    AT(psi, Apsi, nbr, bc, V, h); // do matrix times vector

    // psi * Apsi
    sum = 0;
    for(int i=0; i<V; i++){
        sum += psi[i]*Apsi[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&sum, &buffer, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    sum = buffer;
    
    // maybe need a -0.5 here ???? or a 0.5 ???
    return (2.0 * sum);//*a*a*a; // following MK
//    return (-2.0 * sum);

}

// ----------------------------------------------------------------------------
// Applies matrix times vector
void AT(const vector<double> T, vector<double> & AT,
        const vector< vector<float> > & nbr,
        const vector<int> & bc, const int V, const double h){
//    double a = 0.2581;    
    double diag,offdiag;
    diag = -0.5*(-6.0);
    offdiag = -0.5*(1.0);

    for(int n = 0; n < V ; n++){
        AT[n] = (diag/(h*h)) * T[n]; 
        for(int m=0 ; m < 6; m++){ // m < 7
            // CHANGE TO MINUS MINUS MINUS
            AT[n] = AT[n] + offdiag/(h*h) * T[ nbr[n][m] ];            
        }   
    }
// 1 1/6
    // apply the boundary conditions
   for (int n = 0; n < bc.size(); n++){
        AT[ bc[n] ] = 0.0;    
   }

}

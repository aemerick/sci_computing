// function to determine the boundary grid point locations, set nbr
// for within each processor, and set nbrfc which determines the specific
// boundary grid points on all 6 faces
void set_bc_nbr(vector< vector<float > > & nbr, vector<int > & bc,
                vector< vector<int> > & nbrfc,
                const vector<double > & gridDimensions, 
                const vector<double > & center,
                const vector< vector<int> > & p_xyz, const int rank, 
                const int V, const int GLOBAL_L);

// sets the matrix which translates the lexical order processor number
// to the 3D processor coordinates, and records the adjacent processor
// for each processor rank
void set_p_xyz(vector< vector<int > > & p_xyz, const int N);

// gives lexical processor number (rank) for a px,py,pz
int get_p(const int px, const int py, const int pz,
           const int Nx, const int Ny, const int Nz);

// gives lexical order n (local) for a given (local) x,y,z
int get_n(int x, int y, int z, const vector<double > & gridDimensions);


// ----------------------------------------------------------------------------
// Resiszes a 2D vector with zeros
template <class T> 
void zeros2D(vector< vector<T > > & v, int N1, int N2)
{
    // N1 = number of columns
    // N2 = number of rows
    if(v.size() == N1){
      if(v[0].size() == N2){
        for(int i=0; i<N1; i++){
            for(int j=0; j<N2; j++){
                v[i][j] = 0.0;
            }
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

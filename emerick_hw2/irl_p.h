#include <vector>
using namespace std;

// Function to compute the lanczos
void lanczos(int k1, int k2,
             vector< vector<double> > & v, vector<double> & alpha,
             vector<double> & beta, const vector< vector<float> > & nbr,
             const vector< vector<int> > &nbrfc,
             const vector<int> & bc, const vector< vector<int> > & p_xyz,
             const int V,
             const int bufferSize, const int rank, const int numnodes);

// function to compute matrix times vector
void Av(int i, int j, vector< vector<double> > & v,
        const vector< vector<float> > & nbr,
        const vector<int> & bc, const int V);

// function to determine the boundary grid point locations, set nbr
// for within each processor, and set nbrfc which determines the specific
// boundary grid points on all 6 faces
void set_bc_nbr(vector< vector<float > > & nbr, vector<int > & bc,
                vector< vector<int> > & nbrfc,
                const vector<double > & gridDimensions, 
                const vector<double > & center,
                const vector< vector<int> > & p_xyz, const int rank, 
                const int V);

// sets the matrix which translates the lexical order processor number
// to the 3D processor coordinates, and records the adjacent processor
// for each processor rank
void set_p_xyz(vector< vector<int > > & p_xyz, const int N);

// gives lexical processor number (rank) for a px,py,pz
int get_p(const int px, const int py, const int pz,
           const int Nx, const int Ny, const int Nz);

// Function to share v between adjacent faces on each processor
void share_v(vector< vector<double > > & v, const vector< vector<int > >& p_xyz,
             const vector< vector<int > > & nbrfc,            
             const int kshare, // the kth vector of v to share
             const int V, const int full_buffer_size,
             const int rank, const int numnodes);

// gives lexical order n (local) for a given (local) x,y,z
int get_n(int x, int y, int z, const vector<double > & gridDimensions);

// Wrapper on GSL eigensolver to give eigenvectors in VH and eigenvalues in DH
// of the matrix H with dimensions kpmax * kpmax
void eigensolver(const vector< vector<double > > & H, int kpmax,
                 vector< vector<double > > & VH, vector< vector<double > > & DH);

// Wrapper on GSL qr decomposition to give Q and R of the decomposition of a matrix A
// with dimensions kpmax*kpmax
void qrdecomp(const vector< vector<double > > & Avec, vector< vector<double > > & Qvec,
              vector< vector<double > > & Rvec, int kpmax);


// For an empty vector of vectors, resizes to the desired N1xN2 size and fills
// with zeros. For a vector of vectors of size N1xN2, just fills with zeros
template <class T> 
void zeros2D(vector< vector<T > > & v, int N1, int N2);

// For an empty vector, resizes to N1 and fills with zeros. For a vector of size
// N1, fills with zeros
template <class T>
void zeros1D(vector<T > & v, int N1);

// For two matrices of the same dimensions, copies the orig matrix into copied
/// ellement by element
template <class T>
void matrix_copy(vector< vector<T > > & copied, 
                 const vector< vector<T > > & orig);

// returns the maximum absolute value of any element in matrix x
double max_val_matrix(const vector< vector<double > > & x);


void irl_func(vector< vector<double> > & final_evec,
              vector<double>  & final_eval,
              const int edet,
              const vector< double> & gridvals,
              const vector< vector<float> > & nbr,
              const vector< vector<int >  > & nbrfc,
              const vector< vector<int >  > & p_xyz,
              const vector< int> & bc, const int rank, const int numnodes);


// Function to compute the lanczos
void lanczos(int k1, int k2,
             vector< vector<double> > & v, vector<double> & alpha,
             vector<double> & beta, const vector< vector<float> > & nbr,
             const vector< vector<int> > &nbrfc,
             const vector<int> & bc, const vector< vector<int> > & p_xyz,
             const int V,
             const int bufferSize, const int rank, const int numnodes);

// function to compute matrix times vector
void irl_Av(int i, int j, vector< vector<double> > & v,
        const vector< vector<float> > & nbr,
        const vector<int> & bc, const int V);

// Function to share v between adjacent faces on each processor
void share_v(vector< vector<double > > & v, const vector< vector<int > >& p_xyz,
             const vector< vector<int > > & nbrfc,            
             const int kshare, // the kth vector of v to share
             const int V, const int full_buffer_size,
             const int rank, const int numnodes);

// Wrapper on GSL eigensolver to give eigenvectors in VH and eigenvalues in DH
// of the matrix H with dimensions kpmax * kpmax
void eigensolver(const vector< vector<double > > & H, int kpmax,
                 vector< vector<double > > & VH, vector< vector<double > > & DH);

// Wrapper on GSL qr decomposition to give Q and R of the decomposition of a matrix A
// with dimensions kpmax*kpmax
void qrdecomp(const vector< vector<double > > & Avec, vector< vector<double > > & Qvec,
              vector< vector<double > > & Rvec, int kpmax);

// For two matrices of the same dimensions, copies the orig matrix into copied
/// ellement by element
template <class T>
void matrix_copy(vector< vector<T > > & copied, 
                 const vector< vector<T > > & orig);

// returns the maximum absolute value of any element in matrix x
double max_val_matrix(const vector< vector<double > > & x);





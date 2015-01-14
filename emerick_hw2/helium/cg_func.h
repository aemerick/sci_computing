void cg_func(vector<double> & b, vector<double> & final_U,
             const vector< vector<float> > nbr,
             const vector< vector<int> > nbrfc,
             const vector< vector<int> > p_xyz,
             const vector< int> bc, const double h,
             const int rank, const int numnodes );

void share_p(vector<double > & p, const vector< vector<int > > & p_xyz,
             const vector< vector<int > > & nbrfc,            
             const int V, const int full_buffer_size,
             const int rank, const int numnodes);

void cg_Ap(const vector<double> p, vector<double> & Ap,
           const vector< vector<float> > & nbr,
           const vector<int> & bc, const int V, const double h);



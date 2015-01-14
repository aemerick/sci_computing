void fill_vext_ncomp(vector<double> & Vext, vector<double> & ncomp,
               vector<double> & ncomptot, const vector<double> & gridvals,
               const vector<double> & rloc, const vector< vector<int> > & p_xyz,
               const int rank);

double calculate_T(vector< double> & psi, const vector< vector<int > > & p_xyz,
                   const vector< vector<float> > &nbr,
                   const vector< vector<int > > & nbrfc,
                   const vector<int> & bc,
                   const int full_buffer_size, const double a, const int rank,
                   const int numnodes);


//double erf(const double z);

void AT(const vector<double> T, vector<double> & AT,
        const vector< vector<float> > & nbr,
        const vector<int> & bc, const int V, const double h);

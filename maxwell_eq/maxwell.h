void initial_conditions(vector< vector<double> > & A, vector< vector<double> > & E,
                        const vector<double> & rloc,
                        const vector< vector<int> > & p_xyz,
                        const vector<double> & gridVals,
                        const int rank);

void Av(const vector<double> v, vector<double> & Av,
        const vector< vector<double> > & nbr,
        const vector<int> & bc, const int V, const double h);

void update_A(vector<double> & A, const vector<double> & E,
              const double dt);

void laplacian(const vector<double> v, vector<double> & Av,
        const vector< vector<double> > & nbr,
        const vector<int> & bc, const int V, const double h);

void cn_evolve(vector< vector<double> > & A, vector< vector<double> > & E,
               const double dt, const double dx, const vector<double> & center,
               const vector<double> & gridVals,
               const vector< vector<double> > & nbr,
               const vector< vector<int> > & nbrfc,
               const vector< vector<int> > & p_xyz, 
               const vector<int> & bc,
               const int rank, const int numnodes, const int bufferSize);

void sommerfeld(vector< vector<double> >  & v, const vector< vector<double> > & v_old,
                const vector<int> & bc, 
                const vector< vector<double> > & nbr);

void calculate_B(vector< vector<double> > & B,
                 vector<double> & Benergy,
                 vector< vector<double> > & A,
                 const double h,
                 const vector< vector<double> > & nbr,
                 const vector< vector<int> > & nbrfc,
                 const vector<int> & bc,
                 const vector< vector<int> > & p_xyz,
                 const int rank, const int numnodes, const int buffer_size);

void A_anal(vector< vector<double> > & A, const double & t,
                        const vector<double> & rloc,
                        const vector< vector<int> > & p_xyz,
                        const vector<double> & gridVals,
                        const int rank, const double dt);

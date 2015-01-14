//
//  The following program computes the eigenvalues and eigenvectors of the
//  4-th order Hilbert matrix, H(i,j) = 1/(i + j + 1).
//  From:  http://www.gnu.org/software/gsl/manual/html_node/Eigenvalue-and-Eigenvector-Examples.html

#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>

int
main (void)
{
  double data[] = { 1.0  , 1/2.0, 1/3.0, 1/4.0,
                    1/2.0, 1/3.0, 1/4.0, 1/5.0,
                    1/3.0, 1/4.0, 1/5.0, 1/6.0,
                    1/4.0, 1/5.0, 1/6.0, 1/7.0 };

  gsl_matrix_view m 
    = gsl_matrix_view_array (data, 4, 4);

  gsl_vector *eval = gsl_vector_alloc (4);
  gsl_matrix *evec = gsl_matrix_alloc (4, 4);

  gsl_eigen_symmv_workspace * w = 
    gsl_eigen_symmv_alloc (4);
  
  gsl_eigen_symmv (&m.matrix, eval, evec, w);

  gsl_eigen_symmv_free (w);

  gsl_eigen_symmv_sort (eval, evec, 
                        GSL_EIGEN_SORT_ABS_ASC);
  
  {
    int i;

    for (i = 0; i < 4; i++)
      {
        double eval_i 
           = gsl_vector_get (eval, i);
        gsl_vector_view evec_i 
           = gsl_matrix_column (evec, i);

        printf ("eigenvalue = %g\n", eval_i);
        printf ("eigenvector = \n");
        gsl_vector_fprintf (stdout, 
                            &evec_i.vector, "%g");
      }
  }

  gsl_vector_free (eval);
  gsl_matrix_free (evec);

  return 0;
}


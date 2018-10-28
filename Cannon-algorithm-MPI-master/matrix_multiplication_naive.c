#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double * create_random_mat(int n) {
    double *matrix = malloc( n*n * sizeof(double) );
    for (int i = 0; i < n*n; i++) {
        double random =0;
        matrix[i] = random;
    }
    return matrix;
}

int main(int argc, char *argv[]) {

    double *m_a = NULL;
    double *m_b = NULL;
    double *final_matrix = NULL;

    int p, id;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int n = atoi(argv[1]); //size of the matrix

    /** the master initializes the data **/
    if (!id) {

        if (argc != 2) {
          printf ("Command line: %s <m>\n", argv[0]);
          MPI_Finalize();
          exit (1);
        }

        if (n % p != 0) {
            printf("ERROR: Matrix can not be calculated with this number of tasks.\n");
            MPI_Finalize();
            exit (1);
        }

        m_a[0] = 1;
        m_a[1] = 1;
        m_a[2] = 1;
        m_a[3] = 1;
        m_b[0] = 1;
        m_b[1] = 0;
        m_b[2] = 0;
        m_b[3] = 1;
    }

    // allocate memory for 1D-matrices
    if(!id) {
        final_matrix = malloc( n*n * sizeof(double) );
    } else {
        m_a = malloc( n*n * sizeof(double) );
        m_b = malloc( n*n * sizeof(double) );
    }

    // send 1D matrices to workers
    MPI_Bcast(m_a, n*n , MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_b, n*n , MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // calculate the start- and endrow for worker
    int startrow = id * ( n / p );
    int endrow = ((id + 1) * ( n / p)) -1;

    /* calculate sub matrices */
    int number_of_rows = n/p;
    double *result_matrix = calloc(number_of_rows, sizeof(double));

    int position = 0;

    for (int i = startrow; i <= endrow; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                result_matrix[position] +=
                    m_a[ (i * n + k) ] *
                    m_b[ (k * n + j) ];
            }
            position++;
        }
    }

    free(m_a);
    free(m_b);

    /* collect the results */
    MPI_Gather(result_matrix, number_of_rows, MPI_DOUBLE,
           final_matrix, number_of_rows,  MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /** The master presents the results on the console */
    if (!id){
        int i = 0;
        while (i < n*n) {
            printf("%lf\t", final_matrix[i]);
            i++;
            if (i % n == 0)
                printf("\n");
        }
    }

    free(result_matrix);
    free(final_matrix);

    MPI_Finalize();
    return 0;
}

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN(a,b) ((a)<(b)?(a):(b))

int main (int argc, char *argv[])
{
 int n, id, p;
 MPI_Init (&argc, &argv);
 MPI_Barrier(MPI_COMM_WORLD);
 double elapsed_time = -MPI_Wtime();
 MPI_Comm_rank (MPI_COMM_WORLD, &id);
 MPI_Comm_size (MPI_COMM_WORLD, &p);
 MPI_Status status;

 if (argc != 2) {
   if (!id) printf ("Command line: %s <m>\n", argv[0]);
   MPI_Finalize(); exit (1);
 }

 n = atoi(argv[1]); //size of the matrix

 if (n*n != p) {
   if (!id) printf ("p must be equal to sqr(n)\n");
   MPI_Finalize();
   exit (1);
 }

//if p pas carre parfait ou n pas divisible par sqr(p) on s'arrete

 int i = id/n; //id=n*i+j
 int j = id%n;

 int size_block = n/p;

 int *m_a = (int *) malloc (n*n*sizeof(int));
 int *m_b = (int *) malloc (n*n*sizeof(int));
 //double *final_matrix = NULL;

/*
 if (!id){
   final_matrix = (double *) malloc (n*n*sizeof(int));
 }
 */
 m_a[0] = 1;
 m_a[1] = 3;
 m_a[2] = 5;
 m_a[3] = 7;
 m_b[0] = 1;
 m_b[1] = 0;
 m_b[2] = 0;
 m_b[3] = 1;

 int k = (i + j) % n;
 int a = m_a[n*i+k];
 int b = m_b[n*k+j];
 int c = 0;
 for(int l = 0; l < n; l++){
     c = c + a*b;
     MPI_Send(&a, 1, MPI_INT, i*n+((j+n-1)%n), 1, MPI_COMM_WORLD);
     MPI_Send(&b, 1, MPI_INT, ((i+n-1)%n)*n+j, 1, MPI_COMM_WORLD);
     MPI_Recv(&a, 1, MPI_INT, i*n+((j+1)%n), 1, MPI_COMM_WORLD, &status);
     MPI_Recv(&b, 1, MPI_INT, ((i+1)%n)*n+j, 1, MPI_COMM_WORLD, &status);
     MPI_Barrier(MPI_COMM_WORLD);
 }

 //int count = 5;
 //int globalcount;
 //MPI_Reduce(&globalcount, &count , 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 /*
 if (!id){
   for(int p = 0; p < n*n; p++){
      printf("%d\n",matrix[p]);
   }
   printf ("Total elapsed time: %10.6f\n", elapsed_time);
 }
 */
 printf("je suis le processus %d, mon coeff est %d\n",id,c );

 free(m_a);
 m_a=NULL;
 free(m_b);
 m_b=NULL;

 elapsed_time += MPI_Wtime();
 MPI_Finalize ();
 return 0;
 }

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void update_block(int* c, int* a, int *b, int size_block){
  for (int i=0; i< size_block; i++){
    for (int j=0; j< size_block; j++){
      int tmp=0;
      for (int k=0; k< size_block; k++){
        tmp += a[size_block*i+k] * b[size_block*k+j];
      }
      c[size_block*i+j] += tmp;
    }
  }
}

int mod(int x,int N){
  return (x < 0) ? (x % N + N) : (x % N) ;
}

void mult_mat(int* c, int* a, int *b, int size_block){
  for (int i=0; i< size_block; i++){
    for (int j=0; j< size_block; j++){
      int tmp=0;
      for (int k=0; k< size_block; k++){
        tmp += a[size_block*i+k] * b[size_block*k+j];
      }
      c[size_block*i+j] = tmp;
    }
  }
}


int main (int argc, char *argv[])
{
 int n, id, p;
 MPI_Init (&argc, &argv);
 MPI_Barrier(MPI_COMM_WORLD);
 MPI_Comm_rank (MPI_COMM_WORLD, &id);
 MPI_Comm_size (MPI_COMM_WORLD, &p);
 MPI_Status status;
 if (argc != 2) {
   if (!id) printf ("Command line: %s <m>\n", argv[0]);
   MPI_Finalize(); exit (1);
 }

 n = atoi(argv[1]); //size of the matrix

 if (n % ((int) sqrt(p)) !=0 ) {
   if (!id) printf ("p must be a perfect square and sqr(p) must divide n\n");
   MPI_Finalize();
   exit (1);
 }

  int *m_a = malloc( n*n * sizeof(int) );
  int *m_b = malloc( n*n * sizeof(int) );
  int *final_matrix = NULL;

  // initialize randomily the matrices
  for (int k=0; k<n*n; k++){
    m_a[k]=rand() % 20;
  }
  for (int k=0; k<n*n; k++){
    m_b[k]=rand() % 10;
  }

  if (id==0) {
    final_matrix = malloc( n*n * sizeof(int) );
  }

  MPI_Barrier(MPI_COMM_WORLD);

  double startwtime = MPI_Wtime(), endwtime;

 int size_block = n/sqrt(p);
 int new_n = n/size_block;
 int i = id/new_n; //id=new_n*i+j
 int j = id%new_n;

 // initialize the blocks
 int k = (i + j) % new_n;
 int * a = (int *) malloc (size_block*size_block*sizeof(int));
 for(int l = 0; l < size_block*size_block; l++){
   a[l] = m_a[n * (i*size_block+(l/size_block))  +  k*size_block+(l%size_block)];
 }
 int * b = (int *) malloc (size_block*size_block*sizeof(int));
 for(int l = 0; l < size_block*size_block; l++){
   b[l] = m_b[n * (k*size_block+(l/size_block))  +  j*size_block+(l%size_block)];
 }
 int * c = (int *) malloc (size_block*size_block*sizeof(int));
 for(int l = 0; l < size_block*size_block; l++){
   c[l]=0;
 }

 update_block(c,a,b, size_block);

 for(int l = 0; l < new_n-1 ; l++){
       MPI_Send(a, size_block*size_block, MPI_INT, i*new_n + mod(j+new_n-1, new_n), 1, MPI_COMM_WORLD);
       MPI_Send(b, size_block*size_block, MPI_INT, mod(i+new_n-1,new_n)*new_n + j, 1, MPI_COMM_WORLD);
       MPI_Recv(a, size_block*size_block, MPI_INT,i*new_n + mod(j+1, new_n) , 1, MPI_COMM_WORLD, &status);
       MPI_Recv(b, size_block*size_block, MPI_INT, mod(i+1,new_n)*new_n + j, 1, MPI_COMM_WORLD, &status);

       MPI_Barrier(MPI_COMM_WORLD); // we wait for every process to finish the loop
       update_block(c,a,b, size_block);
  }

  MPI_Gather(c, size_block*size_block, MPI_INT,
         final_matrix, size_block*size_block,  MPI_INT, 0, MPI_COMM_WORLD);

  endwtime = MPI_Wtime();

  if (id==0){
     // we reorganize final_matrix correctly so that we can print it
     int *final_matrix2 = (int *) malloc (n*n*sizeof(int));
     for (int k=0; k < p; k++){
       for(int l = 0; l < size_block*size_block; l++){
         i = k/new_n;
         j = k%new_n;
         final_matrix2[n * (i*size_block+(l/size_block)) + j*size_block+(l%size_block)] = final_matrix[k*size_block*size_block+l];
       }
   }
   // we print the resulting matrix
   int tmp = 0;
   while (tmp < n*n) {
       printf("%d\t", final_matrix2[tmp]);
       tmp++;
       if (tmp % n == 0)
           printf("\n");
   }

   printf("\n");

   double startwtime2 = MPI_Wtime();

   // we compute the multiplication between m_a and m_b in a simple non parallel way
   int *true_matrix = (int *) malloc (n*n*sizeof(int));
   mult_mat(true_matrix, m_a, m_b, n);

   double endwtime2 = MPI_Wtime();

   // we print the resulting matrix
   tmp = 0;
   while (tmp < n*n) {
       printf("%d\t", true_matrix[tmp]);
       tmp++;
       if (tmp % n == 0)
           printf("\n");
   }

   printf("\n");

   free(true_matrix);
   true_matrix=NULL;
   free(final_matrix);
   final_matrix=NULL;
   free(final_matrix2);
   final_matrix2=NULL;

   printf("Cannon algorithm: wall clock time = %f\n", endwtime-startwtime);
   printf("Non-parallel computation: wall clock time = %f\n", endwtime2-startwtime2);
 }

 free(m_a);
 m_a=NULL;
 free(m_b);
 m_b=NULL;
 free(a);
 a=NULL;
 free(b);
 b=NULL;
 free(c);
 c=NULL;

 MPI_Finalize ();
 return 0;
 }

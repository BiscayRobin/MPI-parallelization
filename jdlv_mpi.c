// Jeu de la vie avec sauvegarde de quelques itérations
// compiler avec gcc -O3 -march=native (et -fopenmp si OpenMP souhaité)
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

// hauteur et largeur de la matrice
#define HM 1200
#define LM 800

// nombre total d'itérations
#define ITER 10001
// multiple d'itérations à sauvegarder
#define SAUV 1000

#define DIFFTEMPS(a, b) \
  (((b).tv_sec - (a).tv_sec) + ((b).tv_usec - (a).tv_usec) / 1000000.)

/* tableau de cellules */
typedef char Tab[HM][LM];

// initialisation du tableau de cellules
void init(Tab);

// calcule une nouveau tableau de cellules à partir de l'ancien
// - paramètres : ancien, nouveau
void calcnouv(Tab, Tab);

// variables globales : pas de débordement de pile
Tab t1, t2;
Tab tsauvegarde[1 + ITER / SAUV];

int rank, size;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  struct timeval tv_init, tv_beg, tv_end, tv_save;

  gettimeofday(&tv_init, 0);
  init(t1);
  gettimeofday(&tv_beg, 0);
  MPI_Scatter(t1, (HM / size) * LM, MPI_CHAR, t1 + rank * (HM / size), (HM / size) * LM, MPI_CHAR, 0,
              MPI_COMM_WORLD);
  for (int i = 0; i < ITER; i++)
  {
    if (i % 2 == 0)
      calcnouv(t1, t2);
    else
      calcnouv(t2, t1);
    if (i % SAUV == 0)
    {
      MPI_Gather(t1 + rank * (HM / size), (HM / size) * LM, MPI_CHAR, t1 + rank * (HM / size), (HM / size) * LM, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (!rank)
      {
        printf("sauvegarde (%d)\n", i);
        // copie t1 dans tsauvegarde[i/SAUV]
        for (int x = 0; x < HM; x++)
          for (int y = 0; y < LM; y++)
            tsauvegarde[i / SAUV][x][y] = t1[x][y];
      }
    }
  }

  gettimeofday(&tv_end, 0);
  if (!rank)
  {
    FILE *f = fopen("jdlv.out", "w");
    for (int i = 0; i < ITER; i += SAUV)
    {
      fprintf(f, "------------------ sauvegarde %d ------------------\n", i);
      for (int x = 0; x < HM; x++)
      {
        for (int y = 0; y < LM; y++)
          fprintf(f, tsauvegarde[i / SAUV][x][y] ? "*" : " ");
        fprintf(f, "\n");
      }
    }
    fclose(f);
  }
  gettimeofday(&tv_save, 0);
  if (!rank)
  {
    printf("init : %lf s,", DIFFTEMPS(tv_init, tv_beg));
    printf(" calcul : %lf s,", DIFFTEMPS(tv_beg, tv_end));
    printf(" sauvegarde : %lf s\n", DIFFTEMPS(tv_end, tv_save));
  }
  MPI_Finalize();

  return (0);
}

void init(Tab t)
{
  srand(time(0));
#pragma omp parallel for
  for (int i = 0; i < HM; i++)
  {
    for (int j = 0; j < LM; j++)
    {
      // t[i][j] = rand()%2;
      // t[i][j] = ((i+j)%3==0)?1:0;
      // t[i][j] = (i==0||j==0||i==h-1||j==l-1)?0:1;
      t[i][j] = 0;
    }
  }

  t[10][10] = 1;
  t[10][11] = 1;
  t[10][12] = 1;
  t[9][12] = 1;
  t[8][11] = 1;

  t[55][50] = 1;
  t[54][51] = 1;
  t[54][52] = 1;
  t[55][53] = 1;
  t[56][50] = 1;
  t[56][51] = 1;
  t[56][52] = 1;
}

int nbvois(Tab t, int i, int j)
{
  int n = 0;
  if (i > 0)
  { /* i-1 */
    if (j > 0)
      if (t[i - 1][j - 1])
        n++;
    if (t[i - 1][j])
      n++;
    if (j < LM - 1)
      if (t[i - 1][j + 1])
        n++;
  }
  if (j > 0)
    if (t[i][j - 1])
      n++;
  if (j < LM - 1)
    if (t[i][j + 1])
      n++;
  if (i < HM - 1)
  { /* i+1 */
    if (j > 0)
      if (t[i + 1][j - 1])
        n++;
    if (t[i + 1][j])
      n++;
    if (j < LM - 1)
      if (t[i + 1][j + 1])
        n++;
  }
  return (n);
}

void calcnouv(Tab t, Tab n)
{
 MPI_Request request[4];
 MPI_Status status[4];

  if (rank)
    MPI_Isend(t + (rank * (HM / size)), LM, MPI_CHAR, rank - 1, 0,
              MPI_COMM_WORLD, request);
  if (rank < size - 1)
    MPI_Irecv(t + ((rank + 1) * (HM / size)), LM, MPI_CHAR, rank + 1, 0,
             MPI_COMM_WORLD, request + 1);
  if (rank < size - 1)
    MPI_Isend(t + ((rank + 1) * (HM / size)) - 1, LM, MPI_CHAR, rank + 1, 0,
              MPI_COMM_WORLD, request + 2);
  if (rank)
    MPI_Irecv(t + (rank * (HM / size)) - 1, LM, MPI_CHAR, rank - 1, 0,
             MPI_COMM_WORLD, request + 3);

#pragma omp parallel for
for (int i = rank * (HM / size)+1; i < ((rank + 1) * (HM / size))-1; i++)
    for (int j = 0; j < LM; j++)
    {
      int v = nbvois(t, i, j);
      if (v == 3)
        n[i][j] = 1;
      else if (v == 2)
        n[i][j] = t[i][j];
      else
        n[i][j] = 0;
    }

  if (rank)
   MPI_Wait(request, status);
  if (rank < size - 1)
   MPI_Wait(request + 1, status + 1);
  if (rank < size - 1)
   MPI_Wait(request + 2, status + 2);
  if (rank)
   MPI_Wait(request + 3, status + 3);

#pragma omp parallel for
for (int i = 0; i < size; i++)
    for (int j = 0; j < LM; j++)
    {
      int v = nbvois(t, i* (HM / size), j);
      if (v == 3)
        n[i* (HM / size)][j] = 1;
      else if (v == 2)
        n[i* (HM / size)][j] = t[i* (HM / size)][j];
      else
        n[i* (HM / size)][j] = 0;
      v = nbvois(t, ((i + 1) * (HM / size)) - 1, j);
      if (v == 3)
        n[((i + 1) * (HM / size)) - 1][j] = 1;
      else if (v == 2)
        n[((i + 1) * (HM / size)) - 1][j] = t[((i + 1) * (HM / size)) - 1][j];
      else
        n[((i + 1) * (HM / size)) - 1][j] = 0;
    }

}































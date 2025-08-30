#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define ROWS 50000   // number of samples
#define COLS 340     // number of features
#define MAX_ITER 500
#define EPS 1e-9

double data[ROWS][COLS];

// Generate a large synthetic dataset
void generate_dataset() {
  srand(time(NULL));
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      // Add structured correlation + randomness
      data[i][j] = (rand() % 1000) / 100.0 
                 + 0.5 * (i % 100)   // correlation along samples
                 + 0.2 * j;          // correlation along features
    }
  }
}

// Function to compute mean of each column
void mean_center(double data[ROWS][COLS]) {
  for (int j = 0; j < COLS; j++) {
    double sum = 0.0;
    for (int i = 0; i < ROWS; i++) {
      sum += data[i][j];
    }
    double mean = sum / ROWS;
    for (int i = 0; i < ROWS; i++) {
      data[i][j] -= mean;
    }
  }
}

// Compute covariance matrix (COLS x COLS)
void covariance_matrix(double data[ROWS][COLS], double cov[COLS][COLS]) {
  for (int i = 0; i < COLS; i++) {
    for (int j = i; j < COLS; j++) {
      double sum = 0.0;
      for (int k = 0; k < ROWS; k++) {
        sum += data[k][i] * data[k][j];
      }
      cov[i][j] = cov[j][i] = sum / (ROWS - 1);
    }
  }
}

// TODO:
void PCA(double A[COLS][COLS], double eigenvalues[COLS], double eigenvectors[COLS][COLS]) {

}

int main() {
  double t_start;
  double t_end;

  printf("Generating dataset (%d x %d)...\n", ROWS, COLS);
  generate_dataset();

  printf("Mean-centering data...\n");
  mean_center(data);

  printf("Computing covariance matrix...\n");
  double cov[COLS][COLS];
  t_start = omp_get_wtime();
  covariance_matrix(data, cov);
  t_end = omp_get_wtime();
  printf("Covariance matrix calculation took %.4f\n", t_end - t_start);

  // TODO:
  printf("Find the Principal Components...\n");
  double eigenvalues[COLS];
  double eigenvectors[COLS][COLS];
  t_start = omp_get_wtime();
  PCA(cov, eigenvalues, eigenvectors);
  t_end = omp_get_wtime();
  printf("PCA calculation took %.4f\n", t_end - t_start);

  printf("Top 10 eigenvalues:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f ", eigenvalues[i]);
  }

  return 0;
}

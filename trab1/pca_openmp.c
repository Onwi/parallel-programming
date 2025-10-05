#include <stdio.h>
#include <math.h>
#include <omp.h>

#define ROWS 100000  // number of samples
#define COLS 340     // number of features
#define MAX_ITER 500
#define EPS 1e-9

#define SEED 12345 // Fixed seed for reproducibility

double data[ROWS][COLS];

unsigned int hash(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

/* 
  PARALELIZAÇÃO TEMPORÁRIA PARA GERAÇÃO DE DADOS, PARA FACILITAR TESTES.
  UTILIZAR RAND NOVAMENTE E REMOVER DIRETIVAS PARA REALIZAR OS TESTES
  DO RELATÓRIO.
*/
// Generate a large synthetic dataset
void generate_dataset() {
  // srand(SEED); // is not thread-safe
  
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      unsigned int seed = hash(i * COLS + j + SEED);
      // Add structured correlation + randomness
      data[i][j] = (seed % 1000) / 100.0 
                 + 0.5 * (i % 100)   // correlation along samples
                 + 0.2 * j;          // correlation along features
    }
  }
}

// Function to compute mean of each column
void mean_center(double data[ROWS][COLS]) {
  double means[COLS];

  #pragma omp parallel for
  for (int j = 0; j < COLS; j++) {
    double sum = 0.0;
    for (int i = 0; i < ROWS; i++) {
      sum += data[i][j];
    }
    means[j] = sum / ROWS;
  }

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      data[i][j] -= means[j];
    }
  }

  // #pragma omp parallel for
  // for (int j = 0; j < COLS; j++) {
  //   double sum = 0.0;

  //   #pragma omp parallel for reduction(+:sum)
  //   for (int i = 0; i < ROWS; i++) {
  //     sum += data[i][j];
  //   }
  //   double mean = sum / ROWS;

  //   #pragma omp parallel for
  //   for (int i = 0; i < ROWS; i++) {
  //     data[i][j] -= mean;
  //   }
  // }
}

// Compute covariance matrix (COLS x COLS)
void covariance_matrix(double data[ROWS][COLS], double cov[COLS][COLS]) {
  // dynamic pois a computação é para j >= i, gerando um desbalanceamento da carga
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < COLS; i++) {
    for (int j = i; j < COLS; j++) {
      double sum = 0.0;

      //#pragma omp parallel for reduction(+:sum)
      for (int k = 0; k < ROWS; k++) {
        sum += data[k][i] * data[k][j];
      }
      cov[i][j] = cov[j][i] = sum / (ROWS - 1);
    }
  }
}

// Jacobi method
// src: https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm#Algorithm
void PCA(double A[COLS][COLS], double eigenvalues[COLS], double eigenvectors[COLS][COLS]) {
  // Initialize eigenvectors as identity matrix
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < COLS; i++) {
    for (int j = 0; j < COLS; j++) {
      eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
    }
    // eigenvalues[i] = A[i][i];
  }

  #pragma omp parallel for
  for (int i = 0; i < COLS; i++) {
    eigenvalues[i] = A[i][i];
  }  

  for (int iter = 0; iter < MAX_ITER; iter++) {
    // Find largest off-diagonal element
    int p = 0, q = 1;
    double max_val = fabs(A[p][q]);

    #pragma omp parallel
    {
      // cada thread tem suas variáveis
      int local_p = 0, local_q = 1;
      double local_max = fabs(A[0][1]);
      #pragma omp for collapse(2)
      for (int i = 0; i < COLS; i++) {
        for (int j = i + 1; j < COLS; j++) {
          // if (fabs(A[i][j]) > max_val) {
          if (fabs(A[i][j]) > local_max) {
            local_max = fabs(A[i][j]);
            // max_val = fabs(A[i][j]);
            // p = i; q = j;
            local_p = i; 
            local_q = j;
          }
        }
      }

      // os valores compartilhados são atualizados
      #pragma omp critical
      {
        if (local_max > max_val) {
          max_val = local_max;
          p = local_p;
          q = local_q;
        }
      }
    }

    if (max_val < EPS) break; // converged

    double theta = 0.5 * atan2(2*A[p][q], A[q][q] - A[p][p]);
    double c = cos(theta);
    double s = sin(theta);

    // Update matrix A
    double App = c*c*A[p][p] - 2*s*c*A[p][q] + s*s*A[q][q];
    double Aqq = s*s*A[p][p] + 2*s*c*A[p][q] + c*c*A[q][q];
    A[p][q] = A[q][p] = 0.0;
    A[p][p] = App;
    A[q][q] = Aqq;

    #pragma omp parallel for
    for (int i = 0; i < COLS; i++) {
      if (i != p && i != q) {
        double Aip = c*A[i][p] - s*A[i][q];
        double Aiq = s*A[i][p] + c*A[i][q];
        A[i][p] = A[p][i] = Aip;
        A[i][q] = A[q][i] = Aiq;
      }
    }

    // Update eigenvectors
    #pragma omp parallel for
    for (int i = 0; i < COLS; i++) {
      double vip = c*eigenvectors[i][p] - s*eigenvectors[i][q];
      double viq = s*eigenvectors[i][p] + c*eigenvectors[i][q];
      eigenvectors[i][p] = vip;
      eigenvectors[i][q] = viq;
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < COLS; i++) {
    eigenvalues[i] = A[i][i];
  }
}

int main() {
  double t_start;
  double t_end;

  printf("Generating dataset (%d x %d)...\n", ROWS, COLS);
  generate_dataset();

  t_start = omp_get_wtime();

  printf("Mean-centering data...\n");
  mean_center(data);

  printf("Computing covariance matrix...\n");
  double cov[COLS][COLS];
  
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
  printf("\n");

  return 0;
}

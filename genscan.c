#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

// a += b
void userBinaryOperator(void *input1andOutput, void *input2, int dim)
{
    double *a = (double *) input1andOutput;
    double *b = (double *) input2;
	for (int i = 0; i < dim; ++i) {
    	*(a + i) += *(b + i);
	}		
}

#define ubop(x1,x2, dim) userBinaryOperator( (void *) (x1), (void *) (x2), dim);

void generic_scan_parallel(void *X, int n, size_t l, int dim)
{
	char *c = (char *)X;

    if (n == 1) {
        return;
    }
	
	#pragma omp task
	generic_scan_parallel(c, n/2, l, dim);
	generic_scan_parallel(c + n/2 * l, n - n/2, l, dim);
	#pragma omp taskwait
    
    #pragma omp parallel for
    for (int j = 0; j < n - n/2; ++j) {
        ubop(c + (n/2 + j) * l, c + (n/2 - 1) * l, dim);
    }
}

void generic_scan(void *X, int n, size_t l, int dim) 
{
	#pragma omp parallel
    #pragma omp single
	generic_scan_parallel(X, n, l, dim);
}

int main() {
    //omp_set_num_threads();
	clock_t t;

    double arr[300'000'000];
    int n = 300'000'000;

	t = clock();
    generic_scan(&arr, 1, 1 * sizeof(double), 1);
	t = clock() - t;

	t = clock();
    generic_scan(&arr, n, 1 * sizeof(double), 1);
	t = clock() - t;

	double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds   
    printf("genscan 1D took %f seconds to execute \n", time_taken);	

	double arr2[300'000'000][2][2][2];
	n = 300'000'000;

	t = clock();
	generic_scan(&arr2, 1, 8 * sizeof(double), 8);
	t = clock() - t;

	t = clock();
	generic_scan(&arr2, n, 8 * sizeof(double), 8);
	t = clock() - t;	

	time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds   
    printf("genscan 4D took %f seconds to execute \n", time_taken);

    return 0;
}
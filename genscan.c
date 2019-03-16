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

    double arr[10];
    int n = 10;
	arr[0] = 1;
	arr[1] = 1;
	arr[2] = 1;
	arr[3] = 1;
	arr[4] = 1;
	arr[5] = 1;
	arr[6] = 1;	

	t = clock();
    generic_scan(&arr, 1, 1 * sizeof(double), 1);
	t = clock() - t;

	t = clock();
    generic_scan(&arr, n, 1 * sizeof(double), 1);
	t = clock() - t;

	double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds   
    printf("genscan 1D took %f seconds to execute \n", time_taken);

	for (int i = 0; i < 7; ++i)
		printf("%lf ", arr[i]);
	printf("\n");

	double arr2[10][2][2][2];
	n = 10;
	arr2[0][1][0][1] = 1;
	arr2[1][1][0][1] = 1;
	arr2[2][1][0][1] = 1;
	arr2[3][1][0][1] = 1;
	arr2[4][1][0][1] = 1;
	arr2[5][1][0][1] = 1;
	arr2[6][1][0][1] = 1;

	t = clock();
	generic_scan(&arr2, n, 8 * sizeof(double), 8);
	t = clock() - t;

	for (int i = 0; i < 7; ++i) {
		printf("%lf ", arr2[i][1][0][1]);
	}
	printf("\n");

	time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds   
    printf("genscan 4D took %f seconds to execute \n", time_taken);

    return 0;
}
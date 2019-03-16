#include <stdio.h>
#include <stdlib.h>

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

int main( int argc, char **argv){	
	
	// problem setup
	int n=10;
	if(argc>1) n=atoi(argv[1]);
	int *x = (int *) malloc( sizeof(int)*n) ;
	for(int i=0;i<n;i++) x[i] = i%2;

	// print input
	for(int i=0;i<n;i++) printf("%d ",x[i]); 	printf("\n");

	// scan
	genericScan((void*) x,n,sizeof(int));

	//print output
	for(int i=0;i<n;i++) printf("%d ",x[i]); 	printf("\n");

	// clean up
	free(x);
	return 0;
}
	




	


	




#include <omp.h>
#include <stdlib.h>

// a += b
void userBinaryOperator(void *input1andOutput, void *input2) 
{
    double *a = (double *) input1andOutput;
    double *b = (double *) input2;
    *a += *b;
}

#define ubop(x1,x2) userBinaryOperator( (void *) (x1), (void *) (x2) );

void generic_scan(void *X, int n, size_t l) 
{
	char *c = (char *)X;

    if (n == 1) {
        return;
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            generic_scan(c, n/2, l);
        }
        #pragma omp section
        {
            generic_scan(c + n/2 * l, n - n/2, l);
        }
    }    
    
    #pragma omp parallel for 
    for (int j = 0; j < n/2; ++j) {
        ubop(c + (n/2 + j) * l, c + (n/2 - 1) * l);
    }
}

int main() {
    //omp_set_num_threads();
    double arr[1] = {};

    int n = 1;
    arr[0] = 100;
    
    generic_scan(&arr, n, sizeof(double));
    
    return 0;
}
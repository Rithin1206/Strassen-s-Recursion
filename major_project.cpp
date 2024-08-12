// Name : Rajendra Thottempudi
// UIN : 933004901
// Parallel Computing - Major Project
// Parallelizing Strassen’s Matrix-Multiplication Algorithm

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>  
#include <time.h>
#include <omp.h>
#include <ctime>
#include <cmath>
#include <math.h>

int t_val;

template<typename T>
void clearTheMemory(T** array) {
    delete [] *array;
    delete [] array;
}

void addTheseMatrices(int** A, int** B, int** C, int numRows, int numCols) {
    for(int i = 0; i < numRows; ++i) {
        for(int j =0; j < numCols; j++){
        	C[i][j] = A[i][j] + B[i][j];
        }
    }
}


void copyTheseMatrices(int** dest, int** src, int size, int srcStartRow, int srcStartCol) {
	//seeing the number of times this is run
    for(int i = 0; i < size; ++i) {
        dest[i] = &src[srcStartRow + i][srcStartCol];

    }
}

void subtractTheseMatrices(int **A, int **B, int **C, int numRows, int numCols) {
    for(int i = 0; i < numRows; ++i) {
        for(int j = 0; j < numCols; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

template < typename T >
T **allotMemory(int nr, int nc)
{
    T **ptr = new T*[nr];
    T *curr = new T [nr * nc];
    for(int i = 0; i < nr; i++)
    {
        *(ptr + i) = curr;
         curr += nc;
    }
    return ptr;
}

void matrix_multiplication_serial(int** X, int** Y, int** Z, int size) {
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            Z[i][j] = 0.0;
            for(int k = 0; k < size; ++k) {
                Z[i][j] += X[i][k] * Y[k][j];
            }
        }
    }
}

void multiplyingMatrices(int** X, int** Y, int** Z, int xStart, int xEnd, int yStart, int yEnd, int zStart, int zEnd, int depth) {
    for(int i = xStart; i < xEnd; ++i) {
        for(int j = yStart; j < yEnd; ++j) {
            Z[i][j] = 0.0;
            for(int k = zStart; k < zEnd; ++k) {
                Z[i][j] += X[i][k] * Y[k][j];
            }
        }
    }
}

void multiplyingUsingStrassenMethod(int **X, int **Y, int **Z, int size, int depth){
	//First check for the possibility to use serial matrix multiplication

	if( size <= t_val){

		matrix_multiplication_serial(X, Y , Z, size);
	} else {

		int first = size/2, second = size/2, third = size/2;
		int **M1 = allotMemory< int >(first, second);
        int **M2 = allotMemory< int >(first, second);
        int **M3 = allotMemory< int >(first, second);
        int **M4 = allotMemory< int >(first, second);
        int **M5 = allotMemory< int >(first, second);
        int **M6 = allotMemory< int >(first, second);
        int **M7 = allotMemory< int >(first, second);
        int **A11 = new int*[first];
        int **A12 = new int*[first];
        int **A21 = new int*[first];
        int **A22 = new int*[first];
        int **B11 = new int*[third];
        int **B12 = new int*[third];
        int **B21 = new int*[third];
        int **B22 = new int*[third];
        int **C11 = new int*[first];
        int **C12 = new int*[first];
        int **C21 = new int*[first];
        int **C22 = new int*[first];
        copyTheseMatrices(A11, X, first,  0,  0);
        copyTheseMatrices(A12, X, first, 0, third);
        copyTheseMatrices(A21, X, first, first,  0);
        copyTheseMatrices(A22, X, first, first, third);
        copyTheseMatrices(B11, Y, third,  0,  0);
        copyTheseMatrices(B12, Y, third, 0, second);
        copyTheseMatrices(B21, Y, third, third,  0);
        copyTheseMatrices(B22, Y, third, third, second);
        copyTheseMatrices(C11, Z, first,  0,  0);
        copyTheseMatrices(C12, Z, first,  0, second);
        copyTheseMatrices(C21, Z, first, first,  0);
        copyTheseMatrices(C22, Z, first, first, second);

        int **temp_X_M1 = allotMemory< int >(first, third);
        int **temp_Y_M1 = allotMemory< int >(third, second);
        int **temp_X_M2 = allotMemory< int >(first, third);
        int **temp_Y_M3 = allotMemory< int >(third, second);
        int **temp_Y_M4 = allotMemory< int >(third, second);
        int **temp_X_M5 = allotMemory< int >(first, third);
        int **temp_X_M6 = allotMemory< int >(first, second);
        int **temp_Y_M6 = allotMemory< int >(third, second);
        int **temp_X_M7 = allotMemory< int >(first, third);
        int **temp_Y_M7 = allotMemory< int >(third, second);

#pragma omp task
        {
            addTheseMatrices(A11, A22, temp_X_M1, first, third);
            addTheseMatrices(B11, B22, temp_Y_M1, third, second);
            multiplyingUsingStrassenMethod(temp_X_M1, temp_Y_M1, M1, first, 5);
        }
        
#pragma omp task
        {
            addTheseMatrices(A21, A22, temp_X_M2, first, third);
            multiplyingUsingStrassenMethod(temp_X_M2, B11, M2, first, 5);
        }
        
#pragma omp task
        {
            subtractTheseMatrices(B12, B22, temp_Y_M3, third, second);
            multiplyingUsingStrassenMethod(A11, temp_Y_M3, M3, first, 5);
        }
        
#pragma omp task
        {
            subtractTheseMatrices(B21, B11, temp_Y_M4, third, second);
            multiplyingUsingStrassenMethod(A22, temp_Y_M4, M4, first, 5);
        }
        
#pragma omp task
        {
            addTheseMatrices(A11, A12, temp_X_M5, first, third);
            multiplyingUsingStrassenMethod(temp_X_M5, B22, M5, first, 5);
        }
        
#pragma omp task
        {
            subtractTheseMatrices(A21, A11, temp_X_M6, first, third);
            addTheseMatrices(B11, B12, temp_Y_M6, third, second);
            multiplyingUsingStrassenMethod(temp_X_M6, temp_Y_M6, M6, first, 5);
        }

#pragma omp task
        {
            subtractTheseMatrices(A12, A22, temp_X_M7, first, third);
            addTheseMatrices(B21, B22, temp_Y_M7, third, second);
            multiplyingUsingStrassenMethod(temp_X_M7, temp_Y_M7, M7, first, 5);
        }

#pragma omp taskwait

        //using the formula
        for (int i = 0; i < first; i++)
            for (int j = 0; j < second; j++) {
                C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
                C12[i][j] = M3[i][j] + M5[i][j];
                C21[i][j] = M2[i][j] + M4[i][j];
                C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
            }
        
        clearTheMemory< int >(M1);
        clearTheMemory< int >(M2);
        clearTheMemory< int >(M3);
        clearTheMemory< int >(M4);
        clearTheMemory< int >(M5);
        clearTheMemory< int >(M6);
        clearTheMemory< int >(M7);
      	clearTheMemory< int >(temp_X_M1);
        clearTheMemory< int >(temp_Y_M1);
        clearTheMemory< int >(temp_X_M2);
        clearTheMemory< int >(temp_Y_M3);
        clearTheMemory< int >(temp_Y_M4);
        clearTheMemory< int >(temp_X_M5);
        clearTheMemory< int >(temp_X_M6);
        clearTheMemory< int >(temp_Y_M6);
        clearTheMemory< int >(temp_X_M7);
        clearTheMemory< int >(temp_Y_M7);
        
        delete[] A11; delete[] A12; delete[] A21; delete[] A22;
        delete[] B11; delete[] B12; delete[] B21; delete[] B22;
        delete[] C11; delete[] C12; delete[] C21; delete[] C22;        

	}

}

bool areTheseTwoSame(int **C, int **D, int size){
	bool result = true;

	for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                if(D[i][j] != C[i][j]) {
                    result = false;
                }

    return result;
}

int main(int argc, char* argv[])   
{

  int k = atoi(argv[1]);
  int k_dash = atoi(argv[2]);
  int proc = atoi(argv[3]);
  int size = pow(2,k);  

  if (k_dash <= 1 || k_dash >= k) {
	printf("\n You entered wrong values for k and k_dash \n");
    exit(0);
  }
  t_val = pow(2,(k-k_dash));
  
    
  int **A = allotMemory< int >(size, size);
  int i, j, q, l;
  q = 0;
  l = 0;
  int **B = allotMemory< int >(size, size);
  int **C = allotMemory< int >(size, size);
  int **D = allotMemory< int >(size, size);
  while(craze < 400) craze++;
    
  srand((unsigned)time(NULL));
    for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j){
			B[i][j] = rand() % 50;
			A[i][j] = rand() % 50;

	}

    double start_time, end_time, median_time;
    
    omp_set_dynamic(0);
    omp_set_num_threads(proc);

    start_time = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            multiplyingUsingStrassenMethod(A, B, C, size, 5);
        }
    }
    end_time = omp_get_wtime();

  multiplyingMatrices(A, B, D, 0, size, 0, size, 0, size, 5);
  double execution_time = end_time - start_time;
    

   bool areSame = areTheseTwoSame(C,D, size);
        
    
    if(areSame){
        printf("the input values are : k = %d, matrix size = %d, k' = %d, Threshold size = %d, Number of proc = %d, Exec time = %lf sec \n",  k, size, k_dash, t_val,omp_get_max_threads(),execution_time);
    }
    else{
        printf("Outputs obtained with both methods do not match");
    }

  clearTheMemory< int >(A);
  clearTheMemory< int >(B);
  clearTheMemory< int >(C);
  clearTheMemory< int >(D);
    
  return 0;   
}  






#include <stdio.h>
#include <stdlib.h>

__global__ void imprime(int *d_vetor, int *max_d){
    
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("\nComeço do DEVICE 1...\n");
    
    //__syncthreads();
    if (tid_global <= *max_d){
        d_vetor[tid_global] += 0;
        printf("Na GPU:  %d \n", tid_global);
    }
}

int *alocaLinhas(int *Matriz, int n, int m){
    Matriz = (int *) malloc(sizeof(int) * n*m);
    for (int i = 0; i < n*m; i++)
        Matriz[i] = i;
    return Matriz;
}

// Matriz nxm (3x5)
int main(int argc, char *argv[]){

    printf("Começo do HOST...\n");
    int *Matriz;
    int n, m;
    int *max, *max_d;
    
    int *h_vetor;
    int *d_vetor;

    printf("Digite os numeros de n e m\n");
    scanf("%d %d", &n, &m);

    Matriz = alocaLinhas(Matriz, n ,m);


    max     = (int *) malloc(sizeof(int));
    max_d   = (int *) malloc(sizeof(int));
    h_vetor = (int *) malloc(sizeof(int) * n*m);
    //d_vetor =(int *) malloc(sizeof(int) * n*m);

    *max = (n*m);

    printf("valor do max %d\n", *max);
    

    for (int i=0; i <= m+n; i++){
        h_vetor[i] = i;
    }

    for (int i=0; i <= n+m; i+=n){
        h_vetor[n] = i-n;
    }

    
    //cudaMalloc( (void **)&h_vetor, sizeof(int) * 5);
    cudaMalloc( (void **)&d_vetor, sizeof(int) * n*m);
    cudaMalloc((void **)&max_d, sizeof(int));


    //cudaMemcpy(d_vetor, h_vetor, sizeof(int) * n*m , cudaMemcpyHostToDevice);
    cudaMemcpy(d_vetor, Matriz, sizeof(int) * n*m , cudaMemcpyHostToDevice);
    cudaMemcpy(max_d,   max,     sizeof(int),       cudaMemcpyHostToDevice);


    /*CHAMADA DO KERNEL*/
    imprime<<<n, m>>>( d_vetor, max_d);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy( h_vetor, d_vetor, sizeof(int) *n*m ,cudaMemcpyDeviceToHost);
    
    
    for(int i=0; i<n; i++){
        for (int j=0; j<m; j++){
            printf("%3d ", h_vetor[i+j]);
        }
        printf("\n");
    }

    cudaFree(d_vetor);

    printf("HOST terminado!\n");
    return 0;
    
}
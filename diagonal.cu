#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void imprime(int *d_vetor, int *max_d){
    
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("\nComeço do DEVICE 1...\n");
    
    __syncthreads();

    if (tid_global <= *max_d){
        //d_vetor[tid_global] += 0;
        //printf("Na GPU:  %d \n", tid_global);
        //if (tid_global % 4 == 0)
          //  printf("\n%d seu bloco: %d\n",d_vetor[tid_global], blockIdx.x);
        //else
            //printf("%d seu bloco: %d\n",d_vetor[tid_global], blockIdx.x);
    }
    
}

int *alocaLinhas(int *Matriz, int n, int m){
    Matriz = (int *) malloc(sizeof(int) * n*m);
    for (int i = 0; i < n*m; i++)
        Matriz[i] = i;
    return Matriz;
}


int *pegaDi(int *M, int n, int m, int ini){
    
    int *vec;
    int i, cont =0;
    vec = (int *) malloc( sizeof(int) * (ini+1) );
    // printf("\n valor ini:  %d\n", ini);

    if (vec != NULL)
    {
        // Inicia no comeco do vetor e vai voltando na matriz
        for(int i= ini/*ini*m*/; i < m*n; i+=m-1){
            vec[cont] = M[i];
            //printf("%d \n", M[i]);
            //printf("***pegaDi():  %d \n", vec[cont]);
            cont++;
        }
    }
    else
        perror("nao alocou vetor para diagonal\n");

    return vec;
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

    // for (int i=0; i <= m+n; i++){
    //     h_vetor[i] = i;
    // }

    // for (int i=0; i <= n+m; i+=n){
    //     h_vetor[n] = i-n;
    // }


    /********* preenchendo *********/

    
    //cudaMalloc( (void **)&h_vetor, sizeof(int) * 5);
    cudaMalloc( (void **)&d_vetor, sizeof(int) * n*m);
    cudaMalloc( (void **)&max_d  , sizeof(int));


    //cudaMemcpy(d_vetor, h_vetor, sizeof(int) * n*m , cudaMemcpyHostToDevice);
    cudaMemcpy(d_vetor, Matriz, sizeof(int) * n*m , cudaMemcpyHostToDevice);
    cudaMemcpy(max_d,   max,    sizeof(int),       cudaMemcpyHostToDevice);

   
    /*CHAMADA DO KERNEL*/
    // Para cada diagonal ja separada mandar para a GPU(mandar sempre 2, pois a primeira diagonal que mandar vai escrever na segunda)
    
    // for (int i=2; i< -1+n+m-4 /*numero de diagonais max validas*/; i++){
        imprime<<<n, m>>>( d_vetor, max_d); 
    // }


    printf("\n");
    //cudaDeviceSynchronize();
    
    cudaMemcpy( h_vetor, d_vetor, sizeof(int) *n*m ,cudaMemcpyDeviceToHost);

    int **vec;
    vec = (int **) malloc(sizeof(int *) * n+m-1);   // Alocando as diagonais(vetores)
    if (vec == NULL){
        perror("nao alocou a matriz de diagonais\n");
    }

    // Laço para pegar todas as diagonais que seram usadas no calculo
    // ************************ passou do numero de linhas da da ruim
    for (int i= 0; i < n+m-1; i++){
        vec[i] = pegaDi(h_vetor, n, m, i);
        printf("passou aqui %d \n", i);
    }

    printf("\n == Diagonais == \n");
    for(int i=0; i<n; i++){
        for (int j=0; j<m; j++){
           printf("%3d ", vec[i][j]);
        }
        printf("\n");
    }


    printf("*******************\n");

    // Imprime o valor da celula e a posicao  i e j
    printf("== Matriz Bidimensional ==\n");
    for(int i=0; i<n; i++){
        for (int j=0; j<m; j++){
            printf("%5d/(%d,%d) ", h_vetor[m*i+j], i, j);
        }
        printf("\n");
    }

    // for (int i=0; i < n*m; i++){
    //     printf("%d\n", h_vetor[i]);
    // }

    cudaFree(d_vetor);

    printf("HOST terminado!\n");
    return 0;
    
}

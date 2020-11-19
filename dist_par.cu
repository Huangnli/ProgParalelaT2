#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/time.h>

char *aloca_sequencia(int n)
{
	char *seq;

	seq = (char *) malloc((n + 1) * sizeof(char));
	if (seq == NULL)
	{
		perror("\nErro na alocação de estruturas\n") ;
		exit(1) ;
	}
	return seq;
}

int *aloca_matriz(int totalMatriz)
{
	int *mat;

	mat = (int *) malloc((totalMatriz) * sizeof(int));
	if (mat == NULL)
	{
		perror("\nErro na alocação de estruturas\n") ;
		exit(1) ;
	}
	return mat;
}

void distancia_edicao(int n, int m, char *s, char *r, int *d)
{
	int nADiag,			// Número de anti-diagonais
		tamMaxADiag,	// Tamanho máximo (número máximo de células) da anti-diagonal
		aD,				// Anti-diagonais numeradas de 2 a nADiag + 1
		k, i, j,
		t, a, b, c, min;

	nADiag = n + m - 1;
	tamMaxADiag = n;

	// Para cada anti-diagonal
	for (aD = 2; aD <= nADiag + 1; aD++)
	{
		// Para cada célula da anti-diagonal aD
		for (k = 0; k < tamMaxADiag; k++)
		{
			// Calcula índices i e j da célula (linha e coluna)
			i = n - k;
			j = aD - i;
			
			// Se é uma célula válida
			if (j > 0 && j <= m)
			{
				t = (s[i] != r[j] ? 1 : 0);
				a = d[i][j-1] + 1;
				b = d[i-1][j] + 1;
				c = d[i-1][j-1] + t;
				// Calcula d[i][j] = min(a, b, c)
				if (a < b)
					min = a;
				else
					min = b;
				if (c < min)
					min = c;
				d[i][j] = min;
			}
		}
	}
}

void libera(int n, char *s, char *r, int *d)
{
	int i;

	free(s);
	free(r);
	free(d);
}

__global__ void diagonal_Par(int *d, int *tamanho){
	int idGlobal = blockIdx.x * blockDim.x + threadIdx.x;


	return;
}

int main(int argc, char **argv)
{
	int  n,		// Tamanho da sequência s
		 m,		// Tamanho da sequência r
		 *d,	// Matriz de distâncias com tamanho (n+1)*(m+1)
		 i, j;
	char *s,	// Sequência s de entrada (vetor com tamanho n+1)
		 *r;	// Sequência r de entrada (vetor com tamanho m+1)
	FILE *arqEntrada ;	// Arquivo texto de entrada

	int *d_m;

	if(argc != 2)
	{
		printf("O programa foi executado com argumentos incorretos.\n") ;
		printf("Uso: ./dist_seq <nome arquivo entrada>\n") ;
		exit(1) ;
	}

	// Abre arquivo de entrada
	arqEntrada = fopen(argv[1], "rt") ;

	if (arqEntrada == NULL)
	{
		printf("\nArquivo texto de entrada não encontrado\n") ;
		exit(1) ;
	}

	// Lê tamanho das sequências s e r
	fscanf(arqEntrada, "%d %d", &n, &m) ;

	// Aloca vetores s e r
	s = aloca_sequencia(n);
	r = aloca_sequencia(m);
	// Aloca matriz d
	d = aloca_matriz((n+1)*(m+1));

	// Lê sequências do arquivo de entrada
	s[0] = ' ';
	r[0] = ' ';
	fscanf(arqEntrada, "%s", &(s[1])) ;
	fscanf(arqEntrada, "%s", &(r[1])) ;
	
	// Fecha arquivo de entrada
	fclose(arqEntrada) ;

	struct timeval h_ini, h_fim;
	gettimeofday(&h_ini, 0);

	// Inicializa matriz de distâncias d
	for (i = 0; i <= n; i++)
	{
        d[i] = i;
    }
    
    for (j = 1; j <= m; j++)
	{
		d[(m*j)+j] = j;
	}

	cudaMalloc((void **) &d_m, sizeof(int) * n*m);
	cudaMemcpy(d_m, d, sizeof(d), cudaMemcpyHostToDevice);
	int *max;
	*max = sizeof(int) *n*m;

	// numero de blocos == sequencia S 
	diagonal_Par <<<n, 512 >>>(d_m, max);
	

	// Calcula distância de edição entre sequências s e r, por anti-diagonais
	distancia_edicao(n, m, s, r, d);

	gettimeofday(&h_fim, 0);
	// Tempo de execução na CPU em milissegundos
	long segundos = h_fim.tv_sec - h_ini.tv_sec;
	long microsegundos = h_fim.tv_usec - h_ini.tv_usec;
	double tempo = (segundos * 1e3) + (microsegundos * 1e-3);

	printf("Distância=%d\n", d[n][m]);
	printf("Tempo CPU = %.2fms\n", tempo);

	// Libera vetores s e r e matriz d
	libera(n, s, r, d);

	return 0;
}
/**
 * 
 * @file programa.cu
 * @author João Víctor Zárate, Julio Huang e Ricardo Abreu
 * @brief Trabalho 2 - Programação Paralela
 * @version 1.0
 * @date 2020-11-11
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <iostream>
#include <vector>

//Bibliotecas CUDA Trhust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define tam_var unsigned int

using namespace std;


/**********************************************************************************/
/************* Classe Arquivo *******************/
class Arquivo
{
public:
    Arquivo(char *nome_arq, char op);
    ~Arquivo();
    void Escrever(string texto);
    void Ler();

    vector<char> buffer; //Armazena os caracteres lido do arquivo

private:
    FILE *arquivo; //Arquivo aberto
};

/************ Implementação da classe arquivo **********/

/**
 * @brief Construct a new Arquivo:: Arquivo object.
 * Recebe nome do arquivo, operação que irá realizar e abre o arquivo
 * @param nome_arq ponteiro de char do nome do arquivo
 * @param op operação para realizar (r= read, w=write)
 */
 Arquivo::Arquivo(char *nome_arq, char op)
 {
     switch (op)
     {
     /*Abre o arquivo para leitura*/
     case 'r':
         if ((arquivo = fopen(nome_arq, "r")) == NULL)
         {
             perror("Erro ao abrir arquivo!");
             exit(1);
         }
         break;
         /*Abre o arquivo para escrita*/
     case 'w':
         if ((arquivo = fopen(nome_arq, "w")) == NULL)
         {
             perror("Erro ao abrir arquivo!");
             exit(1);
         }
         break;
 
     default:
         cout << "Erro ao abrir arquivo!";
         break;
     }
 }
 
 /**
  * @brief Destroy the Arquivo:: Arquivo object
  * 
  */
 Arquivo::~Arquivo()
 {
     fclose(arquivo);
 }
 
 /**
  * @brief Realiza a leitura do arquivo aberto,
  * e armazena os dados lidos em um buffer.
  */
 void Arquivo::Ler()
 {
     char c; //Caractere lido
     do
     {
         c = fgetc(arquivo);
         buffer.push_back(c); // Adicionado o caractere lido na última posição do array
     } while (c != EOF);
 }
 
 /**
  * @brief Escreve a saida de texto do resultado em um arquivo.
  * Verificar se está correto
  * 
  * @param texto Texto a ser gravado no arquivo de saida
  */
 void Arquivo::Escrever(std::string texto)
 {
     int tam = texto.size();
     for (int i = 0; i < tam; i++)
     {
         putc(texto[i], arquivo);
     }
 }
/**********************************************************************************/
 /************** Classe CalculoDE ***********************/
 class CalculoDE
{
    private:
    tam_var n; //Tamanho das sequências s
    tam_var m; //Tamanho das sequências r
    std::string s; //Sequências de DNA: cadeias de bases nitrogenadas (A, C, G e T)
    std::string r; //Sequências de DNA: cadeias de bases nitrogenadas (A, C, G e T)

    tam_var salvaInfoInt(std::vector<char> info, tam_var &i);
    std::string salvaInfoString(std::vector<char> info, tam_var &i);
    void CalculaMatrizGPU ();

    public:
    CalculoDE(std::vector<char> info);
    ~CalculoDE();  

    thrust::host_vector<thrust::host_vector<tam_var>> d ; //Matriz
    std::string menorDistEdicao;
    void menorQuantEdicao();
};

/************** Implementação CalculoDE *******************/
/**
 * @brief Construct a new CalculoDE:: CalculoDE object
 * 
 */
 CalculoDE::CalculoDE(std::vector<char> info)
 {
     tam_var p = 0;  //Controle de caractere lido do vetor
     // Salvando tamanho das sequências s e r
     n = salvaInfoInt(info, p);
     m = salvaInfoInt(info, p);
     // salvando sequências do arquivo de entrada
     s = salvaInfoString(info, p);
     r = salvaInfoString(info, p);
 
     //Preenche todas as posições da matriz d com "infinito"(maior valor de long int)
     for (tam_var i = 0; i <= n; i++)
     {
         thrust::host_vector<tam_var> v1;
 
         for (tam_var j = 0; j <= m; j++)
         {
             v1.push_back(INT32_MAX);
         }
         d.push_back(v1);
     }
 
     tam_var numADiag,		// Número de anti-diagonais
          tamMaxADiag,	// Tamanho máximo (número máximo de células) da anti-diagonal
          aD,			// Anti-diagonais numeradas de 2 a numADiag + 1
          k, i, j,
          t, a, b, c, min;
 
     numADiag = n + m - 1;
     tamMaxADiag = n;
 
     // Inicializa matriz de distâncias d
     for (i = 0; i <= n; i++)
     {
         d[i][0] = i;
     }
     for (j = 1; j <= m; j++)
     {
         d[0][j] = j;
     }
 
     // Para cada anti-diagonal
     for (aD = 2; aD <= numADiag + 1; aD++)
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
                 t = (s[i-1] != r[j-1] ? 1 : 0);
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
 
     /*
     printf("%s\n", s.c_str());
     printf("%s\n", r.c_str());
 
     for(i = 0; i <= n; i++) 
     {
         for (j = 0; j <= m; j++)
         {
             printf("%u ", d[i][j]);
         }
         printf("\n");
     }
     */
 }
 
 /**
  * @brief Destroy the CalculoDE:: CalculoDE object
  * 
  */
 CalculoDE::~CalculoDE()
 {
 }
 __global__
 void CalculoDE::CalculaMatrizGPU (){
 
 }
 
 /**
  * @brief Devolver numero que esta no arquivo
  * 
  * @param info Vector<char> com as informções lidas do arquivo
  * @param i Ponteiro com indice da posição do vetor
  * @return numero Dado lido do vetor
  */
 tam_var CalculoDE::salvaInfoInt(std::vector<char> info, tam_var &i)
 {
     std::string *buffer = new std::string(); //String temporaria
     char c;                                  //Varialver auxiliar
     do
     {
         c = info[i];
         buffer->push_back(c);
         i++;
     } while ((c != ' ') && (c != '\n'));
     tam_var dado_int = stoi(*buffer); //Converte a string em int
     delete buffer;                //Deleta string criada
     return dado_int;
 }
 
 /**
  * @brief Devolver string que esta no arquivo
  * 
  * @param info Vector<char> com as informções lidas do arquivo
  * @param i Ponteiro com indice da posição do vetor
  * @return string Dado lido do vetor
  */
 std::string CalculoDE::salvaInfoString(std::vector<char> info, tam_var &i)
 {
     std::string *buffer = new std::string(); //String temporaria
     char c;                                  //Varialver auxiliar
     do
     {
         c = info[i];
         buffer->push_back(c);
         i++;
     } while ((c != ' ') && (c != '\n'));
     std::string dado_string = *buffer;
     delete buffer;                //Deleta string criada
     return dado_string;
 }
 
 void CalculoDE::menorQuantEdicao()
 {
     menorDistEdicao = std::to_string(d[n][m]);
 }

 /**********************************************************************************/
 /***************************** MAIN *************************************************/

int main(int argc, char *argv[])
{
    //Verifica se todos os parametros foram passados
    if (argc < 2)
    {
        cout << "Erro! Deve-se passar 2 argumentos:\n";
        exit(1);
    }

    Arquivo arq_entrada(argv[1], 'r'); //Cria um objeto e abre o arquivo passado como parametro
    arq_entrada.Ler();                 //Le os caracteres do arquivo de entrada

    struct timeval inicio, fim;
	gettimeofday(&inicio, 0);

    CalculoDE calDE(arq_entrada.buffer);
    calDE.menorQuantEdicao();  //Armazena o menor distancia de edicao em uma string

    gettimeofday(&fim, 0);
    // Tempo de execução na CPU em milissegundos
	long segundos = fim.tv_sec - inicio.tv_sec;
	long microsegundos = fim.tv_usec - inicio.tv_usec;
	double tempo = (segundos * 1e3) + (microsegundos * 1e-3);
    //Imprimir menor distancia de edicao
    /*
    Arquivo arq_saida(argv[2], 'w');
    arq_saida.Escrever(calDE.menorDistEdicao);
    */
    printf("%s\n", calDE.menorDistEdicao.c_str());
    printf("Tempo CPU = %.2lums\n", tempo);

    exit(0);
}

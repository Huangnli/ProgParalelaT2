#include <iostream>
#include "Arquivo.h"

using namespace std;

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
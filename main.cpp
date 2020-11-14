/**
 * 
 * @file main.cpp
 * @author João Víctor Zárate, Julio Huang e Ricardo Abreu
 * @brief Trabalho 2 - Programação Paralela
 * @version 1.0
 * @date 2020-11-11
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef ARQUIVO_H
#define ARQUIVO_H

#include <iostream>
#include <sys/time.h>

#include "Arquivo.h"
#include "CalculoDE.h"

using namespace std;

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
    unsigned long microsegundos = fim.tv_usec - inicio.tv_usec;
    //Imprimir menor distancia de edicao
    /*
    Arquivo arq_saida(argv[2], 'w');
    arq_saida.Escrever(calDE.menorDistEdicao);
    */
    printf("%s\n", calDE.menorDistEdicao.c_str());
    printf("Tempo de execução = %.2lums\n", microsegundos);

    exit(0);
}

#endif

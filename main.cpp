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

#include "Arquivo.h"
#include "CalculoDE.h"

using namespace std;

int main(int argc, char *argv[])
{
    //Verifica se todos os parametros foram passados
    if (argc < 3)
    {
        cout << "Erro! Deve-se passar 3 argumentos:\n";
        exit(1);
    }

    Arquivo arq_entrada(argv[1], 'r'); //Cria um objeto e abre o arquivo passado como parametro
    arq_entrada.Ler();                 //Le os caracteres do arquivo de entrada

    CalculoDE calDE(arq_entrada.buffer);
    calDE.menorQuantEdicao();  //Armazena o menor distancia de edicao em uma string

    //Imprimir menor distancia de edicao
    Arquivo arq_saida(argv[2], 'w');
    arq_saida.Escrever(calDE.menorDistEdicao);

    exit(0);
}

#endif

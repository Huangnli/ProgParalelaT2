/**
 * @file CalculoDE.cpp
 * @brief Implementação da classe CalculoDE.h
 * @version 1.0
 * @date 2020-11-11
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#include "CalculoDE.h"

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
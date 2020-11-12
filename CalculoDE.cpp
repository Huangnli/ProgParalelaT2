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
CalculoDE::CalculoDE(tam_var n, tam_var m)
{
    tam_var numADiag,		// Número de anti-diagonais
		 tamMaxADiag,	// Tamanho máximo (número máximo de células) da anti-diagonal
		 aD,			// Anti-diagonais numeradas de 2 a numADiag + 1
		 k, i, j,
		 t, a, b, c, min;

	numADiag = n + m - 1;
	tamMaxADiag = n;

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

/**
 * @brief Destroy the CalculoDE:: CalculoDE object
 * 
 */
CalculoDE::~CalculoDE()
{
}
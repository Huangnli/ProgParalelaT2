#include <iostream>
#include <vector>
#include <string>

#define tam_var unsigned int

class CalculoDE
{
    private:
    tam_var n; //Tamanho das sequências s
    tam_var m; //Tamanho das sequências r
    std::string s; //Sequências de DNA: cadeias de bases nitrogenadas (A, C, G e T)
    std::string r; //Sequências de DNA: cadeias de bases nitrogenadas (A, C, G e T)

    tam_var salvaInfoInt(std::vector<char> info, tam_var &i);
    std::string salvaInfoString(std::vector<char> info, tam_var &i);

    public:
    CalculoDE(std::vector<char> info);
    ~CalculoDE();  

    std::vector<std::vector<tam_var>> d; //Matriz
    std::string menorDistEdicao;
    void menorQuantEdicao();
};
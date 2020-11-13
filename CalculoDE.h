#include <iostream>
#include <vector>
#include <string>

#define tam_var unsigned int

class CalculoDE
{
    private:
    tam_var n; //Quantidade de linhas
    tam_var m; //Quantidade de colunas
    std::string s;
    std::string r;

    tam_var salvaInfoInt(std::vector<char> info, tam_var &i);
    std::string salvaInfoString(std::vector<char> info, tam_var &i);

    public:
    CalculoDE(std::vector<char> info);
    ~CalculoDE();  

    std::vector<std::vector<tam_var>> d; //Matriz
    std::string menorDistEdicao;
    void menorQuantEdicao();
};
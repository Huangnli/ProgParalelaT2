#include <iostream>
#include <vector>

using namespace std;

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
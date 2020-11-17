
###################################################
#                                                 #
# João Víctor Zárate Pereira Araújo               #
# Julio Huang                                     #
# Ricardo Abreu                                   #
# Trabalho 2                                      #
# Professor(a): Nahri Moreano                     #
#                                                 #
###################################################
############### Makefile Trabalho 2  ##############
OBJ = main.o CalculoDE.o Arquivo.o

all: $(OBJ)
	nvcc -arch=sm_30 $(OBJ) -o dist_par

%.o: %.cpp
	nvcc -x cu -arch=sm_30 -I. -dc $< -o $@

clean:
	rm -f *.o

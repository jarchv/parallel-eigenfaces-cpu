CFLAGSLIBS = `pkg-config --cflags --libs opencv`

main:
	g++ main.cpp tools.cpp image.cpp eig.cpp nn.cpp -std=c++11 -o main $< $(CFLAGSLIBS) -fopenmp

exec:
	./main
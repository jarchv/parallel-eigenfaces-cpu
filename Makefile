CFLAGSLIBS = `pkg-config --cflags --libs opencv`

% : %.cpp
	g++ -std=c++11 -o $@ $< $(CFLAGSLIBS) 

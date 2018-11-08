ifeq ($(shell uname -s), Darwin)
CXX = g++-8
else
CXX = g++
endif

CXXFLAGS = -std=c++1z -O3 -march=native -funroll-loops 

.PHONY: all
all: main 

main: main.cpp

.PHONY: clean
clean:
	@rm -rf main *.dat *.eps *.aux *.log *.synctex 

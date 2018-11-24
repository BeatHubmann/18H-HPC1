# TODO a): Set the compiler
ifeq ($(shell uname -s), Darwin)
CXX=g++-8
else
CXX=g++
endif

MPICXX ?= mpic++

CXXFLAGS+=-std=c++11 -Wall -Wpedantic -O3

main: main.o
	$(MPICXX) $(CXXFLAGS) main.o -o main

%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o main

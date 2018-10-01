ifeq ($(shell uname -s),Darwin)
CXX = g++-8
else
CXX = g++
endif

CXXFLAGS = -fopenmp -std=c++11 -O3 -march=native -funroll-loops

.PHONY: all
all: question4

question4: ex01_q4_task_d.cpp 
	$(CXX) $(CXXFLAGS) -o Diffusion1D.out $<

.PHONY: clean
clean:
	@rm Diffusion1D.out

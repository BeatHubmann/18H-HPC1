UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CXX=g++
endif
ifeq ($(UNAME_S),Darwin)
	CXX=g++-8
endif

CPPFLAGS=-O3 -Wall -Wextra -std=c++17 -fopenmp

.PHONY: clean run plot
 
pse2d: pse2d.cpp
	$(CXX) $(CPPFLAGS) $< -o $@

clean:
	rm -rf output
	rm -f pse2d movie2d.mp4

run: pse2d
	mkdir -p output
	rm -f output/*u
	./pse2d

plot:
	python3 ./plot.py

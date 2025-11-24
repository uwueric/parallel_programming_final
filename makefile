CC = mpicc
CFLAGS = -Wall -O3

all: part1 part1plus

part1: part1.c rbuf.c
	gcc $(CFLAGS) -fopenmp -o $@ $^ -lm

part1plus: part1plus.cpp rbuf2.cpp
	g++ $(CFLAGS) -fopenmp -o $@ $^ -lm
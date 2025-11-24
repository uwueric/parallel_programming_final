CC = mpicc
CFLAGS = -Wall -O3

all: part1plus

part1plus: part1plus.cpp rbuf2.cpp
	g++ $(CFLAGS) -fopenmp -o $@ $^ -lm

clean:
	rm part1plus
	rm out_0.txt
	rm out_1.txt
	rm out_2.txt
	rm out_3.txt
	rm out_4.txt
	rm out_5.txt
	rm out_6.txt
	rm out_7.txt
	rm out_8.txt
	rm out_9.txt
	rm out_10.txt
	rm out_11.txt
	rm out_12.txt
	rm out_13.txt
	rm out_14.txt
	rm out_15.txt






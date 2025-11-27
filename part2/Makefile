CC      = mpic++
CFLAGS  = -Wall -O3 -std=c++17 -fopenmp
LDFLAGS = -lm

all: p2

p2: p2.cpp rbuf2.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f p2 out_*.txt

CFLAGS	+= -Wall
CFLAGS	+= -O3
CFLAGS	+= -g2

nbody-par: nbody-par.c
	mpicc $(CFLAGS) -o nbody-par nbody-par.c -g -lm

nbody-seq: nbody-seq.c
	gcc $(CFLAGS) -o nbody-seq nbody-seq.c -lm

all : clean nbody-par nbody-seq

clean:
	rm -f *.o nbody-seq nbody-par *~ *core

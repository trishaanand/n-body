CFLAGS	+= -Wall
CFLAGS	+= -O3
CFLAGS	+= -g2

nbody-par: nbody-par.c
	mpicc $(CFLAGS) -o nbody-par nbody-par.c -g -lm

nbody-seq: nbody-seq.c
	gcc $(CFLAGS) -o nbody-seq nbody-seq.c -lm

nbody-par-tau: nbody-par.c
	/home/ppp1705/2018/tau-2.28/x86_64/bin/tau_cc.sh $(CFLAGS) -o nbody-par-tau nbody-par.c -g -lm

all : clean nbody-par nbody-seq nbody-par-tau

clean:
	rm -f *.o nbody-seq nbody-par *~ *core

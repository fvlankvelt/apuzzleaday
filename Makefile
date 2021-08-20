
all: solve puzzle

puzzle: puzzle.c
	gcc -O3 -o puzzle puzzle.c
#	gcc -g -o puzzle puzzle.c

solve: solve.cu
	nvcc -O3 -o solve solve.cu

clean:
	rm puzzle solve

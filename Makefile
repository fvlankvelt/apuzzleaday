
all: solve puzzle

puzzle: puzzle.c
	gcc -O3 -o puzzle puzzle.c
#	gcc -g -o puzzle puzzle.c

solve: solve.cu
	nvcc -O3 -o solve solve.cu

debug:
	nvcc -DDEBUG -o solve solve.cu

reload_driver:
	sudo rmmod nvidia_uvm
	sudo modprobe nvidia_uvm

clean:
	rm -f puzzle solve

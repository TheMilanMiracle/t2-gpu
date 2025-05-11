N=500
M=1000
STEPS=50

default:
	mkdir -p build
	mkdir -p bin
	cmake -S . -B ./build/
	cmake --build ./build/

seq:
	make
	clear
	./bin/seq ${N} ${M} ${STEPS}

cuda:
	make
	clear
	./bin/cuda-parallel ${N} ${M} ${STEPS}

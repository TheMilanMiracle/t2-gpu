N=1024
M=1024:
STEPS=10
MODE="simple"

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
	./bin/simple-cuda ${N} ${M} ${STEPS} ${MODE}

ifeq ($(CXX),)
        CXX= g++
endif

LIBDIR= $(NEMOOBJ)/acc

all: $(LIBDIR)/PowSphwCut.so

$(LIBDIR)/PowSphwCut.so: PowSphwCut.cc
	$(CXX) -o $@ \
	$< -D_FILE_OFFSET_BITS=64 -march=native -mfpmath=sse -mpreferred-stack-boundary=4 -ggdb3 -Wall -Wextra -Winit-self  -Wshadow -Woverloaded-virtual -Wno-unknown-pragmas -O2 -fPIC -funroll-loops -fforce-addr   -rdynamic   -Llib/ -lfalcON -Lutils/lib -lWDutils.gcc -L/home/bovy/Repos/nemo_cvs/lib -lnemo -ldl -shared

#rm'd from NEMO's make: -Iinc/utils/ -I/home/bovy/Repos/nemo_cvs/inc -I/home/bovy/Repos/nemo_cvs/lib